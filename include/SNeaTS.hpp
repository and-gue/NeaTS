#pragma once

#include "float_pfa.hpp"
#include <array>
#include <tuple>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/util.hpp>
#include "gv_elias_fano.hpp"
#include <ranges>
#include <experimental/simd>
#include <execution>
#include <immintrin.h>
#include <functional>

namespace pfa::sneats {
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
                       bpc * (k - i) + LOG2(_n);
            }
        };

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

            auto apply_simd_linear = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1,
                                        floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(x * t1 + t2));
            };

            auto apply_simd_quadratic = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1,
                                           floatv_simd_t t2) -> intv_simd_t {
                --x;
                return stdx::static_simd_cast<intv_simd_t>(stdx::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_simd_radical = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1,
                                         floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t1 * stdx::sqrt(x + s) + t2));
            };

            auto apply_simd_exponential = [](auto x, auto s, floatv_simd_t t0, floatv_simd_t t1,
                                             floatv_simd_t t2) -> intv_simd_t {
                return stdx::static_simd_cast<intv_simd_t>(stdx::round(t2 * stdx::exp(t1 * x)));
            };

            auto apply_linear = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1,
                                   float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::ceil(x * t1 + t2));
            };

            auto apply_quadratic = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1,
                                      float_scalar_t t2) -> int_scalar_t {
                --x;
                return static_cast<int_scalar_t>(std::ceil(t0 * x * x + t1 * x + t2));
            };

            auto apply_radical = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1,
                                    float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t1 * std::sqrt(x + s) + t2));
            };

            auto apply_exponential = [](auto x, auto s, float_scalar_t t0, float_scalar_t t1,
                                        float_scalar_t t2) -> int_scalar_t {
                return static_cast<int_scalar_t>(std::round(t2 * std::exp(t1 * x)));
            };

            x_t offset_res{0};
            x_t start{0};
            x_t end;

            const floatv_simd_t startv([](int i) { return i + 1; });

            for (auto i_model = 0; i_model < mem_out.size(); ++i_model) {
                auto [bpc, model] = mem_out[i_model];
                end = i_model == (mem_out.size() - 1) ? _n : std::visit(
                        [&](auto &&mo) -> x_t { return mo.get_start(); }, mem_out[i_model + 1].second);

                int64_t eps = (bpc != 0) ? BPC_TO_EPSILON(bpc) + 1 : 0;
                intv_simd_t epsv{eps};
                bits_per_correction[i_model] = bpc;
                starting_positions[i_model] = start;

                auto mt = static_cast<poa_t::approx_fun_t>(std::visit(
                        [&](auto &&mo) -> uint8_t { return (uint8_t) mo.type(); }, model));
                auto t = std::visit([&](auto &&mo) -> auto { return mo.parameters(); }, model);
                auto t1 = std::get<2>(t);
                auto t2 = std::get<3>(t);
                coefficients_t1.emplace_back(t1);
                coefficients_t2.emplace_back(t2);
                x_t s;
                float_scalar_t t0;

                std::size_t num_residuals = end - start;
                intv_simd_t _y, y, error;
                floatv_simd_t sv, t0v, t1v, t2v;

                std::function<intv_simd_t(floatv_simd_t, floatv_simd_t, floatv_simd_t, floatv_simd_t,
                                          floatv_simd_t)> simd_op;
                std::function<int_scalar_t(float_scalar_t, float_scalar_t, float_scalar_t, float_scalar_t,
                                           float_scalar_t)> op;

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
        inline void initial_partitioning(It begin, It end) {
            const auto n = std::distance(begin, end);

            _n = n;
            std::vector<int64_t> distance(n + 1, std::numeric_limits<int64_t>::max());

            auto nrows = std::to_underlying(poa_t::approx_fun_t::COUNT); // number of type models
            auto ncols = max_bpc <= 1 ? size_t{1} : size_t{max_bpc}; // number of

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
            simd_make_residuals(begin);
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
                        const auto r = static_cast<int_scalar_t>(sdsl::bits::read_int(
                                residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc));
                        simd_w[i] = r - eps;
                        offset_res += bpc;
                    }
                    simd_w.copy_to(out_start + j, stdx::element_aligned);
                }

                while (j < num_residuals) {
                    const auto r = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u), offset_res & 0x3F, bpc);
                    *(out_start + j) = static_cast<int_scalar_t>(r) - eps;
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

            constexpr auto np = 8;
            auto i_model{0};
            for (; i_model + np < bits_per_correction.size(); i_model += np) {

                uint8_t _bpc;
                typename poa_t::approx_fun_t mt;
#pragma unroll
                for (std::size_t j{0}; j < np; ++j) {
                    end = *(++it_end);
                    mt = static_cast<poa_t::approx_fun_t>(model_types_0[i_model + j] | (model_types_1[i_model + j] << 1));
                    _bpc = bits_per_correction[i_model + j];
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
                bpc = bits_per_correction[i_model];
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
                auto idx_coefficient_t0 = quad_fun_rank(imt + 1) - 1;
                t0 = coefficients_t0[idx_coefficient_t0];
            } else if ((typename poa_t::approx_fun_t) (type_model) == poa_t::approx_fun_t::Sqrt) {
                auto idx_coefficient_s = (fun_1_rank(imt + 1) - quad_fun_rank(imt + 1)) - 1;
                s = coefficients_s[idx_coefficient_s];
            }

            auto model = poa_t::piecewise_non_linear_approximation::make_fun(
                    (typename poa_t::approx_fun_t) (type_model), start_pos, s, t0, t1, t2);
            const auto idx = offset_residual + bpc * (i - start_pos);

            auto residual = sdsl::bits::read_int(residuals.data() + (idx >> 6u), idx & 0x3F, bpc);
            auto _y = std::visit([&](auto &&mo) { return mo(i + 1); }, model);
            auto y = _y + residual;
            if (bpc != 0) y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);
            return y;
        }

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

        template<size_t M>
        inline auto best_models() {
            // Key: (model_type, bpc) -> Value: length
            std::map<std::pair<uint8_t, uint8_t>, size_t> models;

            for (auto i = 0; i < bits_per_correction.size(); ++i) {
                auto start = starting_positions_ef[i];
                auto end = i == (bits_per_correction.size() - 1) ? _n : starting_positions_ef[i + 1];

                uint8_t bpc = bits_per_correction[i];
                uint8_t mt = model_types_0[i] | (model_types_1[i] << 1);
                auto len = end - start;
                models[std::pair<uint8_t, uint8_t>{mt, bpc}] += len;
            }

            std::array<std::pair<std::pair<uint8_t, uint8_t>, x_t>, M> bm{};
            std::partial_sort_copy(models.begin(), models.end(), bm.begin(), bm.end(),
                                   [](const auto &l, const auto &r) { return l.second > r.second; });

            typename poa_t::vec_pna_t m(M);

            for (auto i{0}; i < M; ++i) {
                auto [mt, bpc] = bm[i].first;
                int64_t epsilon = BPC_TO_EPSILON(bpc);
                m[i] = poa_t::make_model((typename poa_t::approx_fun_t) mt, epsilon);
            }

            return m;
        }

        template<typename It>
        inline void custom_partitioning(It begin, It end, const auto &bm) {
            const auto n = std::distance(begin, end);
            _n = n;
            std::vector<int64_t> distance(n + 1, std::numeric_limits<int64_t>::max());

            const auto nmodels = bm.size();
            std::vector<std::pair<std::make_signed_t<x_t>, std::make_signed_t<x_t>>> frontier(nmodels, {0, 0});

            std::vector<out_t> local_partitions(nmodels);
            std::vector<std::pair<uint8_t, std::unique_ptr<out_t>>> previous(n + 1);

            polygon_t g{};
            distance[0] = 0;

            for (auto k = 0; k < _n; ++k) {
                for (auto im{0}; im < nmodels; ++im) {
                    if (frontier[im].second <= k) {
                        auto t = std::visit([&](auto &&model) -> std::tuple<x_t, x_t, out_t> {
                            return pfa::algorithm::make_segment<poa_t>(model, g, (begin + k), end,
                                                                       frontier[im].second);
                        }, bm[im]);

                        frontier[im].first = std::get<0>(t);
                        frontier[im].second = std::get<1>(t);
                        local_partitions[im] = std::get<2>(t);

                    } else { // relax prefix edge (i, k)
                        auto i = frontier[im].first;
                        auto bpc = pfa::algorithm::epsilon_to_bpc(std::visit([](auto &&mod) -> int64_t {
                            return mod.epsilon;
                        }, bm[im]));
                        auto wik = weight_ik(bm[im], i, k, lossy);

                        if (distance[k] > distance[i] + wik) {
                            distance[k] = distance[i] + wik;
                            previous[k] = std::make_pair(bpc, std::make_unique<out_t>(local_partitions[im]));
                        }
                    }
                }

                for (auto im{0}; im < nmodels; ++im) {
                    auto j = frontier[im].second;
                    auto bpc = pfa::algorithm::epsilon_to_bpc(std::visit([](auto &&mod) -> int64_t {
                        return mod.epsilon;
                    }, bm[im]));
                    auto wkj = weight_ik(bm[im], k, j, lossy);

                    if (distance[j] > distance[k] + wkj) {
                        distance[j] = distance[k] + wkj;
                        std::visit([&](auto &&p) {
                            previous[j] = std::make_pair(bpc, std::make_unique<out_t>(p.copy(k)));
                        }, local_partitions[im]);
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

    template<typename x_t = uint32_t, typename y_t = int64_t, typename poly = double, typename T1 = float, typename T2 = double, typename It, size_t M = 5>
    auto partitioning(It begin, It end, uint8_t bpc, size_t training_size) {
        auto _beg = begin;
        auto _end = _beg + training_size;

        compressor<x_t, y_t, poly, T1, T2> icmpr{bpc};
        icmpr.initial_partitioning(_beg, _end);

        _beg = _end;
        _end = end;
        compressor<x_t, y_t, poly, T1, T2> ccmpr{bpc};
        auto bm = icmpr.template best_models<M>();
        ccmpr.custom_partitioning(_beg, _end, bm);

        return std::make_pair(std::move(icmpr), std::move(ccmpr));
    }
}
