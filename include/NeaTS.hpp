#pragma once

#include "float_pfa.hpp"
#include <array>
#include <tuple>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/util.hpp>
#include "my_elias_fano.hpp"
#include <ranges>
#include <experimental/simd>
#include <execution>
#include <immintrin.h>
#include <functional>
#include <stdfloat>

namespace pfa::neats {
    namespace stdx = std::experimental;

    template<typename x_t = uint32_t, typename y_t = int64_t, typename poly = double, typename T1 = float, typename T2 = double>
    class compressor {
        using poa_t = typename pfa::piecewise_optimal_approximation<x_t, y_t, poly, T1, T2>;
        using polygon_t = poa_t::convex_polygon_t;
        using out_t = poa_t::pna_fun_t;

        using int_scalar_t = y_t;
        using uint_scalar_t = std::make_unsigned_t<int_scalar_t>;
        using float_scalar_t = std::conditional_t<sizeof(y_t) == 4, float, double>;

        using uintv_simd_t = stdx::native_simd<uint_scalar_t>;
        using intv_simd_t = stdx::native_simd<int_scalar_t>;

        using floatv_simd_t = stdx::native_simd<float_scalar_t>;
        static_assert(uintv_simd_t::size() == floatv_simd_t::size());
        static constexpr auto simd_width = uintv_simd_t::size();
        static constexpr auto _simd_width_bit_size = simd_width * sizeof(int_scalar_t) * 8; // 512 bits

        std::vector<std::pair<uint8_t, out_t>> mem_out{};

        bool lossy = false;
        uint8_t max_bpc = 32;
        x_t _n = 0;

        x_t residuals_bit_size = 0;

        MyEliasFano<true> starting_positions_ef;
        sdsl::int_vector<64> residuals;

        MyEliasFano<false> offset_residuals_ef;
        sdsl::int_vector<> bits_per_correction;

        sdsl::bit_vector model_types_0;
        sdsl::bit_vector model_types_1;
        sdsl::bit_vector qbv;

        std::vector<T1> coefficients_t0;
        std::vector<T1> coefficients_t1;
        std::vector<T2> coefficients_t2;
        std::vector<x_t> coefficients_s;

        // each pattern depends on approx_fun_t enum
        sdsl::rank_support_v<1> fun_1_rank;
        sdsl::rank_support_v<1> quad_fun_rank;

    public:

        compressor() = default;

        explicit compressor(auto bpc, bool _lossy = false) : max_bpc{bpc}, lossy{_lossy} {}

        size_t inline weight_ik(auto &&m, auto i = 0, auto k = 0, bool _lossy = false) {
            if (_lossy) {
                return std::visit([](auto &&mo) -> size_t {
                    return std::decay_t<decltype(mo)>::fun_t::lossy_size_in_bits();
                }, m);
            } else {
                auto bpc = pfa::algorithm::epsilon_to_bpc(std::visit([](auto &&mod) -> int64_t {
                    return mod.epsilon;
                }, m));
                return std::visit([](auto &&mo) -> size_t { return std::decay_t<decltype(mo)>::fun_t::size_in_bits(); },
                                  m) +
                       bpc * (k - i); //+ LOG2(_n / 20);
            }
        };

        template<typename It>
        inline void make_residuals(It in_start, It in_end) {
            //sdsl::bit_vector starting_positions_bv(_n, 0);

            auto num_partitions = mem_out.size();
            residuals = sdsl::int_vector<64>(CEIL_UINT_DIV(residuals_bit_size, 64) + 1, 0);
            std::vector<uint64_t> starting_positions(num_partitions, 0);
            //starting_positions = sdsl::bit_vector(_n, 0);
            bits_per_correction = sdsl::int_vector<>(num_partitions, 0);
            model_types_0 = sdsl::bit_vector(num_partitions, 0);
            model_types_1 = sdsl::bit_vector(num_partitions, 0);
            qbv = sdsl::bit_vector(num_partitions, 0);

            std::vector<uint64_t> offset_residuals(num_partitions, 0); // minus one because the first offset is 0
            //auto offset = offset_residuals[0];

            uint64_t offset = 0;
            uint64_t start = 0;
            //offset_residuals[0] = 0;
            for (auto index_model_fun = 0; index_model_fun < mem_out.size(); ++index_model_fun) {
                auto [bpc, model] = mem_out[index_model_fun];
                auto end = index_model_fun == (mem_out.size() - 1) ? _n : std::visit(
                        [&](auto &&mo) -> x_t { return mo.get_start(); }, mem_out[index_model_fun + 1].second);

                int64_t eps = BPC_TO_EPSILON(bpc);
                bits_per_correction[index_model_fun] = bpc;
                starting_positions[index_model_fun] = start;
                for (auto j = start; j < end; ++j) {

                    std::visit([&](auto &&mo) {
                        auto _y = mo(j + 1);
                        auto y = *(in_start + j);
                        auto err = static_cast<y_t>(y - _y);
                        const auto residual = uint64_t(err + (eps + 1));
                        sdsl::bits::write_int(residuals.data() + (offset >> 6u), residual, offset & 0x3F,
                                              bpc);
                        offset += bpc;
                    }, model);
                }

                offset_residuals[index_model_fun] = offset;
                auto mt = std::visit([&](auto &&mo) -> uint8_t { return (uint8_t) mo.type(); }, model);

                auto imt = index_model_fun;
                model_types_0[imt] = mt & 0x1;
                model_types_1[imt] = (mt >> 1) & 0x1;

                std::visit([&](auto &&mo) {
                    auto t = mo.parameters();
                    if (std::get<0>(t).has_value()) {
                        coefficients_s.push_back(std::get<0>(t).value());
                    }

                    if (std::get<1>(t).has_value()) {
                        coefficients_t0.push_back(std::get<1>(t).value());
                        qbv[imt] = 1;
                    }
                    coefficients_t1.push_back(std::get<2>(t));
                    coefficients_t2.push_back(std::get<3>(t));
                }, model);

                start = end;
            }

            //starting_positions_ef = sdsl::sd_vector<>(starting_positions_bv);
            starting_positions_ef = MyEliasFano<true>(starting_positions);
            //offset_residuals_ef = sdsl::sd_vector<>(offset_residuals.begin(), offset_residuals.end());
            offset_residuals_ef = MyEliasFano<false>(offset_residuals);

            //sdsl::util::init_support(offset_residuals_ef_sls, &offset_residuals_ef);

            sdsl::util::bit_compress(bits_per_correction);

            //sdsl::util::init_support(starting_positions_rank, &starting_positions_ef);
            //sdsl::util::init_support(starting_positions_select, &starting_positions_ef);

            //sdsl::util::init_support(starting_positions_rank, &starting_positions);
            //sdsl::util::init_support(starting_positions_select, &starting_positions);


            //sdsl::util::init_support(linear_fun_rank, &model_types);
            sdsl::util::init_support(fun_1_rank, &model_types_1);
            sdsl::util::init_support(quad_fun_rank, &qbv);
            //sdsl::util::init_support(exp_fun_rank, &model_types);

            //sdsl::util::bit_compress(model_types);
            mem_out.clear();
            //mem_out.shrink_to_fit();
        }

        template<typename It>
        inline void simd_make_residuals(It in_data) {
            //sdsl::bit_vector starting_positions_bv(_n, 0);

            auto num_partitions = mem_out.size();
            residuals = sdsl::int_vector<64>(CEIL_UINT_DIV(residuals_bit_size, 64) + 1, 0);
            std::vector<uint64_t> starting_positions(num_partitions, 0);
            //starting_positions = sdsl::bit_vector(_n, 0);
            bits_per_correction = sdsl::int_vector<>(num_partitions, 0);
            model_types_0 = sdsl::bit_vector(num_partitions, 0);
            model_types_1 = sdsl::bit_vector(num_partitions, 0);
            qbv = sdsl::bit_vector(num_partitions, 0);

            std::vector<uint64_t> offset_residuals(num_partitions, 0); // minus one because the first offset is 0
            //auto offset = offset_residuals[0];

            auto apply_simd_linear = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(x * t1 + t2));
            };

            auto apply_simd_quadratic = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                --x;
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_simd_radical = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t1 * stdx::sqrt(x + s) + t2));
            };

            auto apply_simd_exponential = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t2 * stdx::exp(t1 * x)));
            };

            auto apply_linear = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(x * t1 + t2));
            };

            auto apply_quadratic = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                --x;
                return static_cast<int_scalar_t>(std::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_radical = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t1 * std::sqrt(x + s) + t2));
            };

            auto apply_exponential = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t2 * std::exp(t1 * x)));
            };

            x_t offset_res{0};
            x_t start{0};
            x_t end;

            const floatv_simd_t startv([](int i) { return i + 1; });

            for (auto i_model = 0; i_model < mem_out.size(); ++i_model) {
                auto [bpc, model] = mem_out[i_model];
                end = i_model == (mem_out.size() - 1) ? _n : std::visit([&](auto &&mo) -> x_t { return mo.get_start(); }, mem_out[i_model + 1].second);

                int64_t eps = (bpc != 0) ? BPC_TO_EPSILON(bpc) + 1 : 0;
                intv_simd_t epsv{eps};
                bits_per_correction[i_model] = bpc;
                starting_positions[i_model] = start;

                auto mt = static_cast<poa_t::approx_fun_t>(std::visit([&](auto &&mo) -> uint8_t { return (uint8_t) mo.type(); }, model));
                auto t = std::visit([&](auto&& mo) -> auto {return mo.parameters();}, model);
                auto t1 = std::get<2>(t);
                auto t2 = std::get<3>(t);
                coefficients_t1.emplace_back(t1);
                coefficients_t2.emplace_back(t2);
                x_t s;
                float_scalar_t t0;

                std::size_t num_residuals = end - start;
                intv_simd_t _y, y, error;
                floatv_simd_t sv, t0v, t1v, t2v;

                std::function<intv_simd_t(floatv_simd_t, floatv_simd_t, floatv_simd_t, floatv_simd_t, floatv_simd_t)> simd_op;
                std::function<int_scalar_t(float_scalar_t, float_scalar_t, float_scalar_t, float_scalar_t, float_scalar_t)> op;

                t1v = floatv_simd_t{t1};
                t2v = floatv_simd_t{t2};
                switch (mt) {
                    case poa_t::approx_fun_t::Linear: {
                        simd_op = apply_simd_linear;
                        op = apply_linear;
                        break;
                    }
                    case poa_t::approx_fun_t::Quadratic: {
                        simd_op = apply_simd_quadratic;
                        op = apply_quadratic;
                        t0 = std::get<1>(t).value();
                        coefficients_t0.emplace_back(t0);
                        t0v = floatv_simd_t{t0};
                        qbv[i_model] = 1;
                        break;
                    }
                    case poa_t::approx_fun_t::Sqrt : {
                        simd_op = apply_simd_radical;
                        op = apply_radical;
                        s = std::get<0>(t).value();
                        coefficients_s.emplace_back(s);
                        sv = floatv_simd_t{static_cast<float_scalar_t>(s)};
                        break;
                    }
                    case poa_t::approx_fun_t::Exponential : {
                        simd_op = apply_simd_exponential;
                        op = apply_exponential;
                        break;
                    }
                }

                auto j{0};
                for (; j + simd_width <= num_residuals; j += simd_width) {
                    y.copy_from(&(*(in_data + j)), stdx::element_aligned);
                    _y = simd_op(startv + j, sv, t0v, t1v, t2v);
                    error = (y - _y) + epsv;

                    for (auto i{0}; i < simd_width; ++i) {
                        auto err = static_cast<uint64_t>(error[i]);
                        sdsl::bits::write_int(residuals.data() + (offset_res >> 6u), err, offset_res & 0x3F,
                                              bpc);
                        offset_res += bpc;
                    }
                }

                for (; j < num_residuals; ++j) {
                    auto _y_st = op(j + 1, s, t0, t1, t2);
                    auto y_st = *(in_data + j);

                    auto err = static_cast<uint64_t>((y_st - _y_st) + eps);
                    sdsl::bits::write_int(residuals.data() + (offset_res >> 6u), err, offset_res & 0x3F, bpc);
                    offset_res += bpc;
                }

                offset_residuals[i_model] = offset_res;
                model_types_0[i_model] = (uint8_t) mt & 0x1;
                model_types_1[i_model] = ((uint8_t) mt >> 1) & 0x1;
                start = end;
                in_data += num_residuals;
            }

            starting_positions_ef = MyEliasFano<true>(starting_positions);
            offset_residuals_ef = MyEliasFano<false>(offset_residuals);
            sdsl::util::bit_compress(bits_per_correction);

            sdsl::util::init_support(fun_1_rank, &model_types_1);
            sdsl::util::init_support(quad_fun_rank, &qbv);

            mem_out.clear();
        }

        // from begin to end the data is already "normalized" (i.e. > 0)
        template<typename It>
        inline void partitioning(It begin, It end) {
            const auto n = std::distance(begin, end);

            _n = n;
            std::vector<int64_t> distance(n + 1, std::numeric_limits<int64_t>::max());
            auto nrows = std::to_underlying(poa_t::approx_fun_t::COUNT); // cols
            auto ncols = max_bpc <= 1 ? size_t{1} : size_t{max_bpc}; // rows
            auto nmodels = ncols * nrows;
            std::vector<std::pair<std::make_signed_t<x_t>, std::make_signed_t<x_t>>> frontier(nmodels, {0, 0});

            std::vector<out_t> local_partitions(nmodels);
            std::vector<std::pair<uint8_t, std::unique_ptr<out_t>>> previous(n + 1);

            polygon_t g{};
            distance[0] = 0;

            typename poa_t::vec_pna_t m(nmodels);
            // m is a matrix of models of size approx_fun_t::COUNT x max_bpc
            // [[pla-0, pla-2, ..., pla-max_bpc],
            // [pea-0, pea-2, ..., pea-max_bpc],
            // [pqa-0, pqa-2, ..., pqa-max_bpc],
            // [psa-0, psa-2, ..., psa-max_bpc]]

            // bpcs
            for (size_t row = 0; row < nrows; ++row) {
                auto model_type = (typename poa_t::approx_fun_t) (row);
                // model types...
                for (size_t col = 0; col < ncols; ++col) {
                    auto im = col + row * ncols;
                    auto epsilon = static_cast<int64_t>(BPC_TO_EPSILON(col + (col >= 1)));
                    m[im] = poa_t::make_model(model_type, epsilon);
                }
            }

            for (auto k = 0; k < _n; ++k) {
                for (size_t row = 0; row < nrows; ++row) {
                    for (size_t col = 0; col < ncols; ++col) {
                        auto im = col + row * ncols;

                        if (frontier[im].second <= k) {
                            auto t = std::visit([&](auto &&model) -> std::tuple<x_t, x_t, out_t> {
                                return pfa::algorithm::make_segment<poa_t>(model, g, (begin + k), end,
                                                                           frontier[im].second);
                            }, m[im]);

                            frontier[im].first = std::get<0>(t);
                            frontier[im].second = std::get<1>(t);
                            local_partitions[im] = std::get<2>(t);

                        } else { // relax prefix edge (i, k)
                            auto i = frontier[im].first;
                            auto bpc = pfa::algorithm::epsilon_to_bpc(std::visit([](auto &&mod) -> int64_t {
                                return mod.epsilon;
                            }, m[im]));
                            auto wik = weight_ik(m[im], i, k, lossy);

                            if (distance[k] > distance[i] + wik) {
                                distance[k] = distance[i] + wik;
                                previous[k] = std::make_pair(bpc, std::make_unique<out_t>(local_partitions[im]));
                            }
                        }
                    }
                }


                for (size_t row = 0; row < nrows; ++row) {
                    for (size_t col = 0; col < ncols; ++col) {
                        auto im = col + row * ncols;
                        auto j = frontier[im].second;
                        auto bpc = pfa::algorithm::epsilon_to_bpc(std::visit([](auto &&mod) -> int64_t {
                            return mod.epsilon;
                        }, m[im]));
                        auto wkj = weight_ik(m[im], k, j, lossy);

                        if (distance[j] > distance[k] + wkj) {
                            distance[j] = distance[k] + wkj;
                            std::visit([&](auto &&p) {
                                previous[j] = std::make_pair(bpc, std::make_unique<out_t>(p.copy(k)));
                            }, local_partitions[im]);
                        }

                    }
                }
            }

            //auto k = std::visit([](auto &&mo) { return mo.get_start(); }, local_partitions[num_models - 1]);
            auto k = n;
            while (k != 0) {
                auto bpc = previous[k].first;
                auto &f = previous[k].second;
                mem_out.emplace_back(bpc, *f);
                auto kp = std::visit([](auto &&mo) -> x_t { return mo.get_start(); }, *f);
                residuals_bit_size += (k - kp) * bpc;
                k = kp;
            }

            std::reverse(mem_out.begin(), mem_out.end());
            //make_residuals(begin, end);
            simd_make_residuals(begin);
        }

        template<typename It>
        inline void decompress(It out_begin, It out_end) const {
            auto n = std::distance(out_begin, out_end);

            assert(n == _n);

            x_t start = 0;
            uint8_t bpc;
            //auto mt = (uint8_t)(model_types_bv[0]) | ((uint8_t)(model_types_bv[1]) << 1);
            uint32_t offset_res = 0;
            auto offset_coefficients = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients_t0 = 0;

            auto l = bits_per_correction.size();
            auto it_end = starting_positions_ef.at(0);

            for (auto index_model_fun = 0; index_model_fun < l; ++index_model_fun) {
                auto end =
                        index_model_fun == (l - 1) ? n : *(++it_end);//starting_positions_select(index_model_fun + 2);

                //start = starting_positions[index_model_fun];
                bpc = bits_per_correction[index_model_fun];
                auto imt = index_model_fun;
                auto mt = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];
                offset_coefficients++;
                std::optional<x_t> s = std::nullopt;
                std::optional<T1> t0 = std::nullopt;

                if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Sqrt) { // Too arbitrary?
                    s = coefficients_s[offset_coefficients_s++];
                } else if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Quadratic) {
                    t0 = coefficients_t0[offset_coefficients_t0++];
                }

                auto model = poa_t::piecewise_non_linear_approximation::make_fun((typename poa_t::approx_fun_t) (mt),
                                                                                 start, s, t0, t1, t2);
                for (auto j = start; j < end; ++j) {
                    uint64_t residual = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u),
                                                             offset_res & 0x3F, bpc);
                    offset_res += bpc;
                    auto y = std::visit([&](auto &&mo) { return mo(j + 1); }, model);
                    auto _y = y + residual;
                    if (bpc != 0) _y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);

                    *(out_begin + j) = _y;
                }

                start = end;
            }
        }


        template<typename T = int64_t, typename float_scalar_t = std::conditional_t<sizeof(T) == 4, float, double>>
        inline void simd_approximations(float_scalar_t *out) { //, It in_begin, It in_end) {
            namespace stdx = std::experimental;

            const auto approx_v = out;
            const auto v_simd_width = floatv_simd_t{static_cast<float_scalar_t>(simd_width)};

            /**
            //** Only for debugging purposes
            auto check = [](auto approx, auto original_value, int64_t epsilon) {
                auto err = static_cast<int64_t>(original_value - approx);
                if (err > epsilon || err < -(epsilon+1)) {
                    std::cerr << "Error during decompression" << std::endl;
                    std::exit(-1);
                }
            };
            //**/

            //auto apply_linear = [](doublev x, doublev t1, doublev t2) -> doublev {return stdx::ceil(x * t1 + t2);};
            //auto apply_quadratic = [](doublev x, doublev t0, doublev t1, doublev t2) -> doublev {return stdx::ceil(t0 * x * x + t1 * x + t2);};
            //auto apply_sqrt = [](doublev x, doublev s, doublev t1, doublev t2) -> doublev {return stdx::round(t1 * stdx::sqrt(x - s) + t2);};
            //auto apply_exp = [](doublev x, doublev t1, doublev t2) -> doublev {return stdx::ceil(t2 * stdx::exp(t1 * x));};

            auto n = _n;

            x_t start{};
            uint8_t bpc{};
            uint32_t offset_res = 0;
            auto offset_coefficients = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients_t0 = 0;

            auto l = bits_per_correction.size();
            auto it_end = starting_positions_ef.at(0);

            std::array<float_scalar_t, simd_width> x_v = {};
            std::iota(x_v.begin(), x_v.end(), 1);
            //static_assert(x_v[simd_size - 1] == 8.0, "x_v is not initialized correctly");
            //std::vector<max_t> approx_v(n);

            for (auto index_model_fun = 0; index_model_fun < l; ++index_model_fun) {
                auto end = index_model_fun == (l - 1) ? n : *(++it_end);
                bpc = bits_per_correction[index_model_fun];
                auto imt = index_model_fun;
                auto mt = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];
                offset_coefficients++;

                auto t0 = coefficients_t0[offset_coefficients_t0];
                auto s = coefficients_s[offset_coefficients_s];
                auto t0v = floatv_simd_t{static_cast<float_scalar_t>(t0)};
                auto sv = floatv_simd_t{static_cast<float_scalar_t>(s)};

                floatv_simd_t t1v{static_cast<float_scalar_t>(t1)};
                floatv_simd_t t2v{static_cast<float_scalar_t>(t2)};

                auto _iw{0};
                floatv_simd_t _iwv{static_cast<float_scalar_t>(_iw)};
                float_scalar_t eps =
                        bpc != 0 ? static_cast<float_scalar_t>(BPC_TO_EPSILON(bpc) + 1) : float_scalar_t{0};
                //floatv_simd_t epsv{eps};
                floatv_simd_t v, r;

                if (static_cast<poa_t::approx_fun_t>(mt) == poa_t::approx_fun_t::Linear) {
                    auto j{start};
                    for (; j + simd_width <= end; j += simd_width) {
                        // f = apply_linear(_, m, q); apply_simd(j + 1, f) ?
                        v.copy_from(&x_v[0], stdx::element_aligned);
                        r.copy_from(&approx_v[j], stdx::element_aligned);
                        v = v + v_simd_width * _iwv;
                        r = stdx::ceil(t1v * v + t2v) + r;
                        r.copy_to(&approx_v[j], stdx::element_aligned);
                        _iwv = floatv_simd_t{++_iw};
                        //doublev apply_simd([&apply_linear, t1, t2, j](auto i){return apply_linear(j + 1, t1, t2);});
                    }

                    while (j < end) {
                        auto f = std::ceil(t1 * ((j - start) + 1) + t2);
                        approx_v[j] += f;
                        ++j;
                    }

                } else if (static_cast<poa_t::approx_fun_t>(mt) == poa_t::approx_fun_t::Quadratic) {
                    auto j{start};
                    for (; j + simd_width <= end; j += simd_width) {
                        // f = apply_linear(_, m, q); apply_simd(j + 1, f) ?
                        v.copy_from(&x_v[0], stdx::element_aligned);
                        r.copy_from(&approx_v[j], stdx::element_aligned);
                        v = v - 1 + v_simd_width * _iwv;
                        r = stdx::ceil(t0v * v * v + t1v * v + t2v) + r;
                        r.copy_to(&approx_v[j], stdx::element_aligned);
                        _iwv = floatv_simd_t{++_iw};
                        //doublev apply_simd([&apply_linear, t1, t2, j](auto i){return apply_linear(j + 1, t1, t2);});
                    }

                    while (j < end) {
                        auto x = j - start;
                        auto f = std::ceil(t0 * x * x + t1 * x + t2);
                        approx_v[j] += f;
                        ++j;
                    }
                    offset_coefficients_t0++;
                } else if (static_cast<poa_t::approx_fun_t>(mt) == poa_t::approx_fun_t::Sqrt) {
                    auto j{start};
                    for (; j + simd_width <= end; j += simd_width) {
                        v.copy_from(&x_v[0], stdx::element_aligned);
                        r.copy_from(&approx_v[j], stdx::element_aligned);
                        v = v + v_simd_width * _iwv + sv;
                        r = stdx::round(t1v * stdx::sqrt(v) + t2v) + r;
                        r.copy_to(&approx_v[j], stdx::element_aligned);
                        _iwv = floatv_simd_t{++_iw};
                    }

                    while (j < end) {
                        auto x = (j + 1) - (start - s);
                        auto f = std::round(t1 * std::sqrt(x) + t2);
                        approx_v[j] += f;
                        ++j;
                    }

                    offset_coefficients_s++;
                } else if (static_cast<poa_t::approx_fun_t>(mt) == poa_t::approx_fun_t::Exponential) {
                    auto j{start};
                    for (; j + simd_width <= end; j += simd_width) {
                        // f = apply_linear(_, m, q); apply_simd(j + 1, f) ?
                        v.copy_from(&x_v[0], stdx::element_aligned);
                        r.copy_from(&approx_v[j], stdx::element_aligned);
                        v = v + v_simd_width * _iwv;
                        r = stdx::round(t2v * stdx::exp(t1v * v)) + r;
                        r.copy_to(&approx_v[j], stdx::element_aligned);
                        _iwv = floatv_simd_t{++_iw};
                        //doublev apply_simd([&apply_linear, t1, t2, j](auto i){return apply_linear(j + 1, t1, t2);});
                    }

                    while (j < end) {
                        auto x = (j + 1) - start;
                        auto f = std::round(t2 * std::exp(t1 * x));
                        approx_v[j] += f;
                        ++j;
                    }
                }

                /** Only for debugging purposes
                for (auto _i = start; _i < end; ++_i) {
                    assert(_i < n);
                    check(approx_v[_i], *(in_begin + _i), BPC_TO_EPSILON(bpc));
                }
                **/
                start = end;
            }
        }

        /** Extract contiguous bits from a 64-bit integer. */
        inline uint64_t bextr(uint64_t word, unsigned int offset, unsigned int length) const {
#ifdef __BMI__
            return _bextr_u64(word, offset, length);
#else
            return (word >> offset) & sdsl::bits::lo_set[length];
#endif
        }

/** Reads the specified number of bits (must be < 58) from the given position. */
        inline uint64_t read_field(const uint64_t *data, uint64_t bit_offset, uint8_t length) const {
            assert(length < 58);
            auto ptr = reinterpret_cast<const char*>(data);
            auto word = *(reinterpret_cast<const uint64_t *>(ptr + bit_offset / 8));
            return bextr(word, bit_offset % 8, length);
        }

        template<typename T>
        inline void simd_decompress(T *out) {
            auto unpack_residuals = [this](const auto im, x_t offset_res, const auto num_residuals, auto *out_start) {
                constexpr auto _simd_width_bit_size = simd_width * sizeof(int_scalar_t) * 8; // 512 bits
                const uint8_t bpc = bits_per_correction[im];
                // NOTE: we are assuming bpc != 0
                const int_scalar_t eps = BPC_TO_EPSILON(bpc) + 1;

                auto j{0};
                intv_simd_t simd_w{};
                for (; j + simd_width <= num_residuals; j += simd_width) {
                    for (std::size_t i{0}; i < simd_width; ++i) {
                        const auto r = static_cast<int_scalar_t>(read_field(residuals.data(), offset_res, bpc));
                        //const auto r = static_cast<int_scalar_t>(sdsl::bits::read_int(
                        //        residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc));
                        simd_w[i] = r - eps;
                        offset_res += bpc;
                    }
                    simd_w.copy_to(out_start + j, stdx::element_aligned);
                }

                while (j < num_residuals) {
                    const auto r = static_cast<int_scalar_t>(read_field(residuals.data(), offset_res, bpc));
                    //const auto r = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc);
                    *(out_start + j) = r - eps;
                    offset_res += bpc;
                    ++j;
                }
            };

            auto apply_simd_linear = [](auto x, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(x * t1 + t2));
            };

            auto apply_simd_quadratic = [](auto x, floatv_simd_t t0, floatv_simd_t t1,
                                           floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_simd_radical = [](auto x, auto s, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t1 * stdx::sqrt(x + s) + t2));
            };

            auto apply_simd_exponential = [](auto x, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t2 * stdx::exp(t1 * x)));
            };

            auto apply_linear = [](auto x, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(x * t1 + t2));
            };

            auto apply_quadratic = [](auto x, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_radical = [](auto x, auto s, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t1 * std::sqrt(x + s) + t2));
            };

            auto apply_exponential = [](auto x, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t2 * std::exp(t1 * x)));
            };

            const floatv_simd_t startv([](int i) { return i + 1; });
            const floatv_simd_t qstartv([](int i) { return i; });
            auto unpack_poa = [&](poa_t::approx_fun_t mt, x_t offset_coeff_s, x_t offset_coeff_t0, x_t offset_coeff,
                                  const auto num_residuals, auto *out_start) {

                float_scalar_t t0, t1, t2, s;
                floatv_simd_t t0v, t1v, t2v, sv;
                intv_simd_t _residuals{};
                switch (mt) {
                    case poa_t::approx_fun_t::Linear : {
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_linear(startv + j, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_linear(j + 1, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                    case poa_t::approx_fun_t::Quadratic : {
                        t0 = coefficients_t0[offset_coeff_t0];
                        t0v = floatv_simd_t{t0};
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_quadratic(qstartv + j, t0v, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_quadratic(j, t0, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                    case poa_t::approx_fun_t::Exponential : {

                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_exponential(startv + j, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_exponential(j + 1, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                    case poa_t::approx_fun_t::Sqrt : {
                        s = static_cast<float_scalar_t>(coefficients_s[offset_coeff_s]);
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};
                        sv = floatv_simd_t{s};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_radical(startv + j, sv, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_radical(j + 1, s, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                }
            };

            uint8_t bpc{};
            x_t offset_res{0};
            auto it_end = starting_positions_ef.at(0);
            x_t offset_coefficients{0};
            x_t offset_coefficients_s{0};
            x_t offset_coefficients_t0{0};

            x_t start{0};
            x_t end;
            const auto bpc_width = bits_per_correction.width();

            constexpr auto np = 8;
            auto i_model{0};
            for (; i_model + np < bits_per_correction.size(); i_model += np) {

                uint8_t _bpc;
                typename poa_t::approx_fun_t mt;
#pragma unroll
                for (std::size_t j{0}; j < np; ++j) {
                    end = *(++it_end);
                    mt = static_cast<poa_t::approx_fun_t>(model_types_0[i_model + j] | (model_types_1[i_model + j] << 1));
                    //_bpc = bits_per_correction[i_model + j];
                    _bpc = read_field(bits_per_correction.data(), (i_model + j) * bpc_width, bpc_width);
                    if (_bpc != 0) unpack_residuals(i_model + j, offset_res, end - start, out + start);
                    unpack_poa(mt, offset_coefficients_s, offset_coefficients_t0, offset_coefficients + j, end - start,
                               out + start);

                    offset_coefficients_s += mt == poa_t::approx_fun_t::Sqrt;
                    offset_coefficients_t0 += mt == poa_t::approx_fun_t::Quadratic;

                    offset_res += _bpc * (end - start);
                    start = end;
                }
                offset_coefficients += np;
            }

            for (; i_model < bits_per_correction.size(); ++i_model) {
                end = i_model == (bits_per_correction.size() - 1) ? _n : *(++it_end);
                bpc = read_field(bits_per_correction.data(), i_model * bpc_width, bpc_width);
                auto mt = static_cast<poa_t::approx_fun_t>(model_types_0[i_model] | (model_types_1[i_model] << 1));
                if (bpc != 0) unpack_residuals(i_model, offset_res, end - start, out + start);
                unpack_poa(mt, offset_coefficients_s, offset_coefficients_t0, offset_coefficients, end - start,
                           out + start);
                //unpack_pla(offset_coefficients, end - start, out + start);
                offset_coefficients++;
                offset_coefficients_s += mt == poa_t::approx_fun_t::Sqrt;
                offset_coefficients_t0 += mt == poa_t::approx_fun_t::Quadratic;
                offset_res += bpc * (end - start);
                start = end;
            }
        }

        template<typename T>
        inline void simd_scan(x_t s, x_t e, T *out) const {
            auto unpack_residuals = [this](const auto im, x_t offset_res, const auto num_residuals, auto *out_start) {
                constexpr auto _simd_width_bit_size = simd_width * sizeof(int_scalar_t) * 8; // 512 bits
                const uint8_t bpc = bits_per_correction[im];
                // NOTE: we are assuming bpc != 0
                const int_scalar_t eps = BPC_TO_EPSILON(bpc) + 1;

                auto j{0};
                intv_simd_t simd_w{};
                for (; j + simd_width <= num_residuals; j += simd_width) {
                    for (std::size_t i{0}; i < simd_width; ++i) {
                        const auto r = static_cast<int_scalar_t>(read_field(residuals.data(), offset_res, bpc));
                        //const auto r = static_cast<int_scalar_t>(sdsl::bits::read_int(
                        //        residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc));
                        simd_w[i] = r - eps;
                        offset_res += bpc;
                    }
                    simd_w.copy_to(out_start + j, stdx::element_aligned);
                }

                while (j < num_residuals) {
                    const auto r = static_cast<int_scalar_t>(read_field(residuals.data(), offset_res, bpc));
                    //const auto r = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc);
                    *(out_start + j) = r - eps;
                    offset_res += bpc;
                    ++j;
                }
            };

            auto apply_simd_linear = [](auto x, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(x * t1 + t2));
            };

            auto apply_simd_quadratic = [](auto x, floatv_simd_t t0, floatv_simd_t t1,
                                           floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_simd_radical = [](auto x, auto s, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t1 * stdx::sqrt(x + s) + t2));
            };

            auto apply_simd_exponential = [](auto x, floatv_simd_t t1, floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t2 * stdx::exp(t1 * x)));
            };

            auto apply_linear = [](auto x, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(x * t1 + t2));
            };

            auto apply_quadratic = [](auto x, float_scalar_t t0, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_radical = [](auto x, auto s, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t1 * std::sqrt(x + s) + t2));
            };

            auto apply_exponential = [](auto x, float_scalar_t t1, float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t2 * std::exp(t1 * x)));
            };

            const floatv_simd_t startv([](int i) { return i + 1; });
            const floatv_simd_t qstartv([](int i) { return i; });
            auto unpack_poa = [&](poa_t::approx_fun_t mt, x_t offset_coeff_s, x_t offset_coeff_t0, x_t offset_coeff,
                                  x_t st_off, const auto num_residuals, auto *out_start) {

                float_scalar_t t0, t1, t2, s;
                floatv_simd_t t0v, t1v, t2v, sv;
                intv_simd_t _residuals{};
                switch (mt) {
                    case poa_t::approx_fun_t::Linear : {
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_linear(startv + j + st_off, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_linear(j + st_off + 1, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Quadratic : {
                        t0 = coefficients_t0[offset_coeff_t0];
                        t0v = floatv_simd_t{t0};
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_quadratic(qstartv + j + st_off, t0v, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_quadratic(j + st_off, t0, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                    case poa_t::approx_fun_t::Exponential : {

                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_exponential(startv + j + st_off, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_exponential(j + 1 + st_off, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                    case poa_t::approx_fun_t::Sqrt : {
                        s = static_cast<float_scalar_t>(coefficients_s[offset_coeff_s]);
                        t1 = coefficients_t1[offset_coeff];
                        t2 = coefficients_t2[offset_coeff];
                        t1v = floatv_simd_t{t1};
                        t2v = floatv_simd_t{t2};
                        sv = floatv_simd_t{s};

                        auto j{0};
                        for (; j + simd_width <= num_residuals; j += simd_width) {
                            _residuals.copy_from(out_start + j, stdx::element_aligned);
                            _residuals += apply_simd_radical(startv + j + st_off, sv, t1v, t2v);
                            _residuals.copy_to(out_start + j, stdx::element_aligned);
                        }

                        for (; j < num_residuals; ++j) {
                            int_scalar_t _y = apply_radical(j + 1 + st_off, s, t1, t2);
                            *(out_start + j) += _y;
                        }
                        break;
                    }
                }
            };

            auto pre = starting_positions_ef.predecessor(s);
            auto imt = pre.index();
            uint64_t start = *pre;
            uint64_t st_off = s - start;
            //uint64_t end_pos = e;
            //uint8_t bpc = bits_per_correction[imt];
            auto offset_res = imt == 0 ? 0 : offset_residuals_ef[imt - 1];

            auto it_end = pre;
            typename poa_t::approx_fun_t mt;
            auto offset_coefficients = imt;
            auto offset_coefficients_t0 = quad_fun_rank(imt);
            auto offset_coefficients_s = fun_1_rank(imt) - quad_fun_rank(imt);

            const auto bpc_width = bits_per_correction.width();
            auto em = starting_positions_ef.predecessor(e).index() + 1;
            uint32_t wp = 0;
            constexpr auto np = 8;
            for (; imt + np < em; imt += np) {
#pragma unroll
                for (std::size_t j{0}; j < np; ++j) {
                    //x_t end = std::min(*(++it_end), (uint64_t) e);
                    x_t end = *(++it_end);
                    mt = static_cast<poa_t::approx_fun_t>(model_types_0[imt + j] | (model_types_1[imt + j] << 1));
                    //_bpc = bits_per_correction[i_model + j];
                    auto _bpc = read_field(bits_per_correction.data(), (imt + j) * bpc_width, bpc_width);
                    if (_bpc != 0)
                        unpack_residuals(imt + j, offset_res + (st_off * _bpc), end - (start + st_off), out + wp);
                    unpack_poa(mt, offset_coefficients_s, offset_coefficients_t0, offset_coefficients + j, st_off,
                               end - (start + st_off), out + wp);

                    offset_coefficients_s += mt == poa_t::approx_fun_t::Sqrt;
                    offset_coefficients_t0 += mt == poa_t::approx_fun_t::Quadratic;

                    wp += end - (start + st_off);
                    offset_res += _bpc * (end - start);
                    start = end;
                    st_off = 0;
                }
                offset_coefficients += np;
            }

            for (; imt < em; ++imt) {
                x_t end = imt == (em - 1) ? e : *(++it_end);
                mt = static_cast<poa_t::approx_fun_t>(model_types_0[imt] | (model_types_1[imt] << 1));
                //_bpc = bits_per_correction[i_model + j];
                auto _bpc = read_field(bits_per_correction.data(), imt * bpc_width, bpc_width);
                if (_bpc != 0) unpack_residuals(imt, offset_res + (st_off * _bpc), end - (start + st_off), out + wp);
                unpack_poa(mt, offset_coefficients_s, offset_coefficients_t0, offset_coefficients, st_off, end - (start + st_off), out + wp);

                offset_coefficients_s += mt == poa_t::approx_fun_t::Sqrt;
                offset_coefficients_t0 += mt == poa_t::approx_fun_t::Quadratic;

                wp += end - (start + st_off);
                offset_res += _bpc * (end - start);
                start = end;
                st_off = 0;
                offset_coefficients++;
            }
        }


//        template<typename T>
//        inline auto simd_decompress() {
//        namespace stdx = std::experimental;
//        using simd_t = stdx::native_simd<T>;
//        //using vector_simd_t = std::vector<simd_t>;
//        constexpr auto simd_width = simd_t::size();
//
//
//
//            auto approximations = simd_approximations(); // vector<max_t>
//            auto residuals = simd_residuals(); // vector<max_t>
//
//            /****
//             * #include <immintrin.h>
//
//            // Extract contiguous bits from a 64-bit integer.
//            inline uint64_t bextr(uint64_t word, unsigned int offset, unsigned int length) {
//                #ifdef __BMI__
//                return _bextr_u64(word, offset, length);
//                #else
//                return (word >> offset) & sdsl::bits::lo_set[length];
//                // #endif
//            }
//
//            // Reads the specified number of bits (must be < 58) from the given position.
//            inline uint64_t read_field(const uint64_t *data, uint64_t bit_offset, uint8_t length) {
//                assert(length < 58);
//                auto ptr = reinterpret_cast<const char*>(data);
//                auto word = *(reinterpret_cast<const uint64_t *>(ptr + bit_offset / 8));
//                return bextr(word, bit_offset % 8, length);
//            }
//            ****/
//
//            auto num_models{bits_per_correction.size()};
//            x_t start{};
//            uint8_t bpc{};
//            uint32_t offset_res{};
//            auto it_end = starting_positions_ef.at(0);
//            for (auto im = 0; im < num_models; ++im) {
//                auto end = im == (num_models - 1) ? _n : *(++it_end);
//                bpc = bits_per_correction[im];
//                auto j{start};
//                auto eps = bpc != 0? static_cast<max_t>(BPC_TO_EPSILON(bpc) + 1) :max_t{0};
//                auto epsv = doublev_t{eps};
//                for (; j + simd_size <= end; j += simd_size) {
//                    doublev_t residuals_v([__residual_at, offset_res, bpc](auto i){return __residual_at(offset_res + (i * bpc), bpc);});
//                    doublev_t v(&approximations[j], stdx::element_aligned);
//                    v = v + residuals_v;
//                    v -= epsv; // not when bpc == 0
//                    v.copy_to(&approximations[j], stdx::element_aligned);
//                    offset_res += bpc * simd_size;
//                }
//
//                while (j < end) {
//                    auto residual = __residual_at(offset_res, bpc);
//                    approximations[j] += residual;
//                    approximations[j] -= eps;
//                    ++j;
//                    offset_res += bpc;
//                }
//
//                start = end;
//            }
//            return approximations;
//        }

        /* really slow
        inline auto simd_residuals() {
            namespace stdx = std::experimental;
            using max_t = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>;

            using doublev_t = stdx::native_simd<max_t>;
            constexpr std::size_t simd_size = doublev_t::size();
            constexpr doublev_t simd_sizev{static_cast<max_t>(simd_size)};

            auto approximations = simd_approximations(); // vector<max_t>

            auto __residual_at = [this](auto off, auto bpc) -> max_t {
                return static_cast<max_t>(sdsl::bits::read_int(residuals.data() + (off >> 6u),
                                                                        off & 0x3F, bpc));
            };

            auto num_models{bits_per_correction.size()};
            x_t start{};
            uint8_t bpc{};
            uint32_t offset_res{};
            auto it_end = starting_positions_ef.at(0);
            for (auto im = 0; im < num_models; ++im) {
                auto end = im == (num_models - 1) ? _n : *(++it_end);
                bpc = bits_per_correction[im];
                auto j{start};
                auto eps = bpc != 0? static_cast<max_t>(BPC_TO_EPSILON(bpc) + 1) :max_t{0};
                auto epsv = doublev_t{eps};
                for (; j + simd_size <= end; j += simd_size) {
                    doublev_t residuals_v([__residual_at, offset_res, bpc](auto i){return __residual_at(offset_res + (i * bpc), bpc);});
                    doublev_t v(&approximations[j], stdx::element_aligned);
                    v = v + residuals_v;
                    v -= epsv; // not when bpc == 0
                    v.copy_to(&approximations[j], stdx::element_aligned);
                    offset_res += bpc * simd_size;
                }

                while (j < end) {
                    auto residual = __residual_at(offset_res, bpc);
                    approximations[j] += residual;
                    approximations[j] -= eps;
                    ++j;
                    offset_res += bpc;
                }

                start = end;
            }
            return approximations;
        }
        */


        size_t size_in_bits() const {
            return sizeof(*this) * 8 + residuals.bit_size() +  // offset_residuals.size() * sizeof(uint32_t) * 8 +
                   //sdsl::size_in_bytes(offset_residuals_ef) * 8 +
                   starting_positions_ef.size_in_bytes() * 8 +
                   sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                   sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() +
                   model_types_0.bit_size() + model_types_1.bit_size() + qbv.bit_size() +
                   offset_residuals_ef.size_in_bytes() * 8 + starting_positions_ef.size_in_bytes() * 8 +
                   //(sdsl::size_in_bytes(offset_residuals_ef_sls) + sdsl::size_in_bytes(starting_positions_select) +
                   // sdsl::size_in_bytes(starting_positions_rank) +
                   (sdsl::size_in_bytes(fun_1_rank) + sdsl::size_in_bytes(quad_fun_rank)) * 8 +
                   //sdsl::size_in_bytes(starting_positions_ef) * 8 +
                   bits_per_correction.bit_size();
        }

        size_t storage_size_in_bits() const {
            auto num_partitions = bits_per_correction.size();
            return residuals.bit_size() + sizeof(x_t) * 8 * coefficients_s.size() +
                   sizeof(T1) * 8 * coefficients_t0.size() +
                   sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() +
                   model_types_0.bit_size() + model_types_1.bit_size() + (num_partitions * sizeof(x_t) * 8) +
                   bits_per_correction.bit_size();
        }

        size_t coefficients_size_in_bits() const {
            return sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                   sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size();
        }

        size_t residuals_size_in_bits() const {
            return residuals.bit_size();
        }

        void size_info(bool header = true) const {
            if (header) {
                std::cout
                        << "residuals,offset_residuals,coefficients,model_types,rank_supports,starting_positions,bits_per_correction,meta\n";
            }
            std::cout << residuals.bit_size() << ",";
            //std::cout << offset_residuals.size() * sizeof(uint32_t) * 8 << ",";
            std::cout << offset_residuals_ef.size_in_bytes() * 8 << ",";
            std::cout << sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                         sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() << ",";
            std::cout << model_types_0.bit_size() + model_types_1.bit_size() + qbv.bit_size() << ",";
            //std::cout << (sdsl::size_in_bytes(offset_residuals_ef_sls) +
            //             sdsl::size_in_bytes(starting_positions_select) +
            //              sdsl::size_in_bytes(starting_positions_rank) +
            std::cout << (sdsl::size_in_bytes(fun_1_rank) + sdsl::size_in_bytes(quad_fun_rank)) * 8 << ",";
            std::cout << starting_positions_ef.size_in_bytes() * 8 << ",";
            std::cout << bits_per_correction.bit_size() << ",";
            std::cout << sizeof(*this) * 8 << std::endl;
        }

        void storage_size_info() const {
            auto num_partitions = model_types_0.size();
            std::cout << residuals.bit_size() << ","
                      << sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                         sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() << ","
                      << model_types_0.bit_size() + model_types_1.bit_size() << ","
                      << num_partitions * sizeof(x_t) * 8 << ","
                      << bits_per_correction.bit_size() << ",";
        }

        constexpr inline y_t operator[](x_t i) const {
            auto res = starting_positions_ef.predecessor(i);
            auto index_model = res.index();
            //auto index_model = it_model.index();
            //auto start_pos = static_cast<x_t>(starting_positions_select(index_model + 1));
            //auto start_pos = *it_model;
            uint64_t start_pos = *res;

            auto imt = index_model;
            auto type_model = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);
            auto bpc = bits_per_correction[index_model];
            //auto offset_residual =
            //        index_model == 0 ? 0 : offset_residuals_ef_sls(index_model);//offset_residuals_ef[index_model - 1];
            auto offset_residual = index_model == 0 ? 0 : offset_residuals_ef[index_model - 1];

            auto t1 = coefficients_t1[index_model];
            auto t2 = coefficients_t2[index_model];

            std::optional<x_t> s = std::nullopt;
            std::optional<T1> t0 = std::nullopt;

            if ((typename poa_t::approx_fun_t) (type_model) == poa_t::approx_fun_t::Quadratic) {
                //auto idx_coefficient_t0 = quad_fun_rank(imt + 1) - 1;
                auto idx_coefficient_t0 = quad_fun_rank(imt);
                t0 = coefficients_t0[idx_coefficient_t0];
            } else if ((typename poa_t::approx_fun_t) (type_model) == poa_t::approx_fun_t::Sqrt) {
                //auto idx_coefficient_s = (fun_1_rank(imt + 1) - quad_fun_rank(imt + 1)) - 1;
                auto idx_coefficient_s = fun_1_rank(imt) - quad_fun_rank(imt);
                s = coefficients_s[idx_coefficient_s];
            }

            auto model = poa_t::piecewise_non_linear_approximation::make_fun(
                    (typename poa_t::approx_fun_t) (type_model), start_pos, s, t0, t1, t2);
            const auto idx = offset_residual + bpc * (i - start_pos);

            auto residual = static_cast<y_t>(sdsl::bits::read_int(residuals.data() + (idx >> 6u), idx & 0x3F, bpc));
            if (bpc != 0) residual -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);

            auto _y = std::visit([&](auto &&mo) { return mo(i + 1); }, model);
            //y_t residual = read_field(residuals.data(), offset_residual + bpc * (i - start_pos), bpc);
            auto y = _y + residual;
            return y;
        }


        /*
        template<typename It>
        inline void print_info(It in_begin, It in_end, It out_begin, It out_end) const {
            //assert(std::distance(begin, end) == mem_out.size());
            auto n = std::distance(in_begin, in_end);

            auto mean_linear_length = 0;
            auto mean_nonlinear_length = 0;
            auto num_linear_models = 0;
            auto num_nonlinear_models = 0;
            size_t bit_size = 0;

            std::cout << "#models: " << mem_out.size() << std::endl;

            size_t start = 0;
            for (auto index_model = 0; index_model < mem_out.size(); ++index_model) {
                auto [bpc, model] = mem_out[index_model];
                auto end = index_model == (mem_out.size() - 1) ? n : std::visit(
                        [&](auto &&mo) -> x_t { return mo.get_start(); }, mem_out[index_model + 1].second);

                bit_size += bpc * (end - start);
                int64_t eps = BPC_TO_EPSILON(bpc);

                std::visit([&](auto &&mo) {
                    bit_size += mo.size_in_bits();
                    auto t = mo.type();
                    if (t == 0) {
                        mean_linear_length += (end - start);
                        ++num_linear_models;
                    } else {
                        mean_nonlinear_length += (end - start);
                        ++num_nonlinear_models;
                    }
                }, model);

                for (auto j = start; j < end; ++j) {

                    std::visit([&](auto &&mo) {
                        auto _y = mo(j + 1);
                        *(out_begin + j) = _y;

                        auto y = *(in_begin + j);
                        auto err = static_cast<y_t>(y - _y);
                        if ((err > 0 && err > eps) || (err < 0 && err < (-eps - 1))) {
                            std::cout << "error: " << err << " > " << eps << std::endl;
                            std::cout << "j: " << j << ", y: " << y << ", _y: " << _y << std::endl;
                            std::cout << "something wrong! decompress failed" << std::endl;
                            exit(1);
                        }

                    }, model);
                }

                start = end;
            }

            auto byte_size = bit_size / 8.0;
            std::cout << "#linear models: " << num_linear_models << ", #non-linear models: " << num_nonlinear_models << std::endl;
            std::cout << "mean linear models length: " << mean_linear_length / (double) num_linear_models << ", mean non-linear models length: " << mean_nonlinear_length / (double) num_nonlinear_models << std::endl;
            std::cout << "decompressed size: " << _n * sizeof(y_t) << " bytes | compressed size: " << byte_size << " bytes" << std::endl;
            std::cout << "compression ratio: " << byte_size / ((double) _n * sizeof(y_t)) << std::endl;
            std::cout << "mem_out size (num models): " << mem_out.size() << std::endl;
            std::cout << "correction bit size: " << residuals_bit_size << std::endl;
            std::cout << "correction size: " << residuals.bit_size() << std::endl;
        }
        */

        constexpr auto size() const {
            return _n;
        }

        constexpr uint8_t bits_per_residual() {
            return max_bpc;
        }

        inline size_t serialize(std::ostream &os, sdsl::structure_tree_node *v = nullptr,
                                const std::string &name = "") const {
            if (_n == 0) {
                throw std::runtime_error("compressor empty");
            }

            auto child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
            size_t written_bytes = 0;
            written_bytes += sdsl::write_member(max_bpc, os, child, "max_bpc");
            written_bytes += sdsl::write_member(_n, os, child, "size");
            written_bytes += sdsl::write_member(residuals_bit_size, os, child, "residuals_bit_size");

            written_bytes += starting_positions_ef.serialize(os, child, "starting_positions_ef");

            written_bytes += sdsl::serialize(residuals, os, child, "residuals");
            written_bytes += offset_residuals_ef.serialize(os, child, "offset_residuals_ef");
            written_bytes += sdsl::serialize(bits_per_correction, os, child, "bits_per_correction");
            written_bytes += sdsl::serialize(model_types_0, os, child, "model_types_0");
            written_bytes += sdsl::serialize(model_types_1, os, child, "model_types_1");
            written_bytes += sdsl::serialize(qbv, os, child, "qbv");

            written_bytes += sdsl::write_member(coefficients_t0.size(), os, child, "coefficients_t0.size()");
            written_bytes += sdsl::serialize_vector(coefficients_t0, os, child, "coefficients_t0");

            written_bytes += sdsl::write_member(coefficients_s.size(), os, child, "coefficients_s.size()");
            written_bytes += sdsl::serialize_vector(coefficients_s, os, child, "coefficients_s");

            written_bytes += sdsl::write_member(coefficients_t1.size(), os, child, "coefficients_t1.size()");
            written_bytes += sdsl::serialize_vector(coefficients_t1, os, child, "coefficients_t1");
            written_bytes += sdsl::serialize_vector(coefficients_t2, os, child, "coefficients_t2");

            sdsl::structure_tree::add_size(child, written_bytes);
            return written_bytes;
        }

        /*
        void to_csv(std::string filename) const {
            assert(!mem_out.empty());
            std::ofstream fout(filename);
            fout << std::setprecision(16);
            fout << "model_type,start,bpc,c0,c1,c2,residuals_size,plx,ply,prx,pry" << std::endl;

            auto start = 0;
            for (auto index_model_fun = 0; index_model_fun < mem_out.size(); ++index_model_fun) {
                auto end = index_model_fun == (mem_out.size() - 1) ? _n : std::visit(
                        [&](auto &&mo) -> x_t { return mo.get_start(); }, mem_out[index_model_fun + 1].second);
                auto [bpc, model] = mem_out[index_model_fun];

                std::string t0_str = "";
                std::string t1_str;
                std::string t2_str;
                std::string mt;
                std::string start_str = std::to_string(start);
                std::string bpc_str = std::to_string(bpc);
                std::string residuals_size = std::to_string(bpc * (end - start));
                std::visit([&](auto &&mo) {
                    auto t = mo.parameters();
                    if (std::get<1>(t).has_value()) {
                        t0_str = std::to_string(std::get<1>(t).value());
                    }
                    t1_str = std::to_string(std::get<2>(t));
                    t2_str = std::to_string(std::get<3>(t));

                    mt = std::to_string((uint8_t) mo.type());
                    auto d = mo.diagonal();

                    fout << mt << "," << start_str << ","
                         << bpc_str << "," << t0_str << "," << t1_str << "," << t2_str << "," << residuals_size << ","
                         << d.p0().x() << "," << d.p0().y() << "," << d.p1().x() << "," << d.p1().y() << std::endl;

                }, model);
                start = end;
            }
        }
        */

        void inline write_info_csv(std::ostream &ostream) {
            ostream.precision(5);
            ostream << std::fixed;
            ostream << "ifragment,bpc,type,s,t0,t1,t2,len,residuals_uint32" << std::endl;
            x_t start = 0;
            uint8_t bpc;
            //auto mt = (uint8_t)(model_types_bv[0]) | ((uint8_t)(model_types_bv[1]) << 1);
            uint32_t offset_res = 0;
            auto offset_coefficients = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients_t0 = 0;

            auto l = bits_per_correction.size();
            auto it_end = starting_positions_ef.at(0);

            for (auto index_model_fun = 0; index_model_fun < l; ++index_model_fun) {
                auto end =
                        index_model_fun == (l - 1) ? _n : *(++it_end);//starting_positions_select(index_model_fun + 2);
                ostream << index_model_fun << ",";

                //start = starting_positions[index_model_fun];
                bpc = bits_per_correction[index_model_fun];
                ostream << (uint64_t) (bpc) << ",";
                auto imt = index_model_fun;
                auto mt = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);
                ostream << (uint64_t) (mt) << ",";

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];
                offset_coefficients++;
                std::optional<x_t> s = std::nullopt;
                std::optional<T1> t0 = std::nullopt;

                if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Sqrt) { // Too arbitrary?
                    s = coefficients_s[offset_coefficients_s++];
                    ostream << (uint64_t) (s.value()) << ",";
                } else if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Quadratic) {
                    t0 = coefficients_t0[offset_coefficients_t0++];
                    ostream << "," << (std::float64_t) (t0.value());
                } else {
                    ostream << ",";
                }
                ostream << "," << (std::float64_t) (t1) << "," << (std::float64_t) (t2) << ",";
                ostream << (end - start) << ",";
                auto model = poa_t::piecewise_non_linear_approximation::make_fun((typename poa_t::approx_fun_t) (mt),
                                                                                 start, s, t0, t1, t2);
                std::stringstream residual_str;
                for (auto j = start; j < end; ++j) {
                    uint64_t residual = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u),
                                                             offset_res & 0x3F, bpc);
                    if (j == start) residual_str << "[";
                    residual_str << residual;
                    if (j == (end - 1)) residual_str << "]";
                    else residual_str << ":";
                    offset_res += bpc;
                }
                ostream << residual_str.str() << std::endl;
                start = end;
            }
        }

        static auto load(std::istream &is) {
            decltype(max_bpc) _max_bpc = 0;
            sdsl::read_member(_max_bpc, is);
            compressor<x_t, y_t, poly, T1, T2> lc{_max_bpc};
            sdsl::read_member(lc._n, is);
            sdsl::read_member(lc.residuals_bit_size, is);

            lc.starting_positions_ef.load(is);

            sdsl::load(lc.residuals, is);
            lc.offset_residuals_ef.load(is);
            sdsl::load(lc.bits_per_correction, is);
            sdsl::load(lc.model_types_0, is);
            sdsl::load(lc.model_types_1, is);
            sdsl::load(lc.qbv, is);

            size_t coefficients_t0_size;
            sdsl::read_member(coefficients_t0_size, is);
            lc.coefficients_t0 = decltype(coefficients_t0)(coefficients_t0_size);
            sdsl::load_vector<T1>(lc.coefficients_t0, is);

            size_t coefficients_s_size;
            sdsl::read_member(coefficients_s_size, is);
            lc.coefficients_s = decltype(lc.coefficients_s)(coefficients_s_size);
            sdsl::load_vector<x_t>(lc.coefficients_s, is);

            size_t coefficients_t1_size;
            sdsl::read_member(coefficients_t1_size, is);
            lc.coefficients_t1 = decltype(coefficients_t1)(coefficients_t1_size);
            sdsl::load_vector<T1>(lc.coefficients_t1, is);
            lc.coefficients_t2 = decltype(coefficients_t2)(coefficients_t1_size);
            sdsl::load_vector<T2>(lc.coefficients_t2, is);

            sdsl::util::init_support(lc.fun_1_rank, &lc.model_types_1);
            sdsl::util::init_support(lc.quad_fun_rank, &lc.qbv);

            return lc;
        }


    };
}