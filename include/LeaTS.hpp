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

namespace pfa::leats {
    namespace stdx = std::experimental;

    template<typename x_t = uint32_t, typename y_t = int64_t, typename poly_t = double, typename T1 = float, typename T2 = double>
    class compressor {
        using poa_t = typename pfa::piecewise_optimal_approximation<x_t, y_t, poly_t, T1, T2>;
        using pla_t = poa_t::piecewise_linear_approximation;
        using polygon_t = poa_t::convex_polygon_t;
        using out_t = typename pla_t::fun_t;

        using int_scalar_t = y_t;
        using uint_scalar_t = std::make_unsigned_t<int_scalar_t>;
        using float_scalar_t = std::conditional_t<sizeof(y_t) == 4, float, double>;

        using uintv_simd_t = stdx::native_simd<uint_scalar_t>;
        using intv_simd_t = stdx::native_simd<int_scalar_t>;

        using floatv_simd_t = stdx::native_simd<float_scalar_t>;
        static_assert(uintv_simd_t::size() == floatv_simd_t::size());
        static constexpr auto simd_width = uintv_simd_t::size();
        static constexpr auto _simd_width_bit_size = simd_width * sizeof(int_scalar_t) * 8; // 512 bits


        // vector of pairs (bpc, linear_model)
        std::vector<std::pair<uint8_t, out_t>> mem_out{};

        uint8_t max_bpc = 32;
        x_t _n = 0;
        x_t residuals_bit_size = 0;

        MyEliasFano<true> starting_positions_ef;
        sdsl::int_vector<64> residuals;

        MyEliasFano<false> offset_residuals_ef;
        sdsl::int_vector<> bits_per_correction;

        std::vector<T1> coefficients_t1;
        std::vector<T2> coefficients_t2;

    public:

        compressor() = default;

        explicit compressor(auto bpc) : max_bpc{bpc} {}

        template<typename It>
        inline void make_residuals(It in_data) {
            auto num_partitions = mem_out.size();
            residuals = sdsl::int_vector<64>(CEIL_UINT_DIV(residuals_bit_size, 64) + 1, 0);
            std::vector<uint64_t> starting_positions(num_partitions, 0);
            bits_per_correction = sdsl::int_vector<>(num_partitions, 0);

            std::vector<uint64_t> offset_residuals(num_partitions, 0);

            x_t offset_res{0};
            x_t start{0};
            x_t end;

            const floatv_simd_t startv([](int i) { return i + 1; });

            for (auto i_model = 0; i_model < mem_out.size(); ++i_model) {
                auto [bpc, model] = mem_out[i_model];
                end = i_model == (mem_out.size() - 1) ? _n : mem_out[i_model + 1].second.get_start();

                int_scalar_t eps = (bpc != 0) ? BPC_TO_EPSILON(bpc) + 1 : 0;
                intv_simd_t epsv{eps};
                bits_per_correction[i_model] = bpc;
                starting_positions[i_model] = start;
                auto t = model.parameters();
                auto t1 = std::get<2>(t);
                auto t2 = std::get<3>(t);
                coefficients_t1.emplace_back(t1);
                coefficients_t2.emplace_back(t2);

                std::size_t num_residuals = end - start;
                intv_simd_t _y, y, error;
                auto t1v = floatv_simd_t{t1};
                auto t2v = floatv_simd_t{t2};

                auto j{0};
                for (; j + simd_width <= num_residuals; j += simd_width) {
                    y.copy_from(&(*(in_data + j)), stdx::element_aligned);
                    _y = stdx::static_simd_cast<intv_simd_t>(stdx::ceil(t1v * (startv + j) + t2v));
                    error = (y - _y) + epsv;

                    for (auto i{0}; i < simd_width; ++i) {
                        auto err = static_cast<uint64_t>(error[i]);
                        sdsl::bits::write_int(residuals.data() + (offset_res >> 6u), err, offset_res & 0x3F,
                                              bpc);
                        offset_res += bpc;
                    }
                }

                for (; j < num_residuals; ++j) {
                    auto _y_st = std::ceil((j + 1) * std::get<2>(t) + std::get<3>(t));
                    auto y_st = *(in_data + j);

                    auto err = static_cast<uint64_t>((y_st - _y_st) + eps);
                    sdsl::bits::write_int(residuals.data() + (offset_res >> 6u), err, offset_res & 0x3F, bpc);
                    offset_res += bpc;
                }

                start = end;
                offset_residuals[i_model] = offset_res;
                in_data = in_data + num_residuals;
            }

            starting_positions_ef = MyEliasFano<true>(starting_positions);
            offset_residuals_ef = MyEliasFano<false>(offset_residuals);

            sdsl::util::bit_compress(bits_per_correction);
            mem_out.clear();
        }

        template<typename It>
        inline void partitioning(It begin, It end) {
            const auto n = std::distance(begin, end);

            _n = n;
            std::vector<int64_t> distance(n + 1, std::numeric_limits<int64_t>::max());
            auto num_linear_models = max_bpc <= 1 ? size_t{1} : size_t{max_bpc}; // rows
            std::vector<std::pair<std::make_signed_t<x_t>, std::make_signed_t<x_t>>> frontier(num_linear_models,
                                                                                              {0, 0});

            std::vector<out_t> local_partitions(num_linear_models);
            std::vector<std::pair<uint8_t, std::unique_ptr<out_t>>> previous(n + 1);

            polygon_t g{};
            distance[0] = 0;

            std::vector<pla_t> m(num_linear_models);

            for (size_t i_model = 0; i_model < num_linear_models; ++i_model) {
                auto epsilon = static_cast<int64_t>(BPC_TO_EPSILON(i_model + (i_model >= 1)));
                m[i_model] = pla_t{epsilon};
            }

            for (auto k = 0; k < _n; ++k) {
                for (size_t im = 0; im < num_linear_models; ++im) {
                    if (frontier[im].second <= k) {
                        auto t = pfa::algorithm::make_segment<poa_t>(m[im], g, (begin + k), end, frontier[im].second);
                        frontier[im].first = std::get<0>(t);
                        frontier[im].second = std::get<1>(t);
                        local_partitions[im] = std::get<2>(t);

                    } else { // relax prefix edge (i, k)
                        auto i = frontier[im].first;
                        auto bpc = pfa::algorithm::epsilon_to_bpc(m[im].epsilon);
                        auto wik = (k - i) * bpc + (sizeof(T1) + sizeof(T2)) * 8; //weight_ik(m[im], i, k, lossy);

                        if (distance[k] > distance[i] + wik) {
                            distance[k] = distance[i] + wik;
                            previous[k] = std::make_pair(bpc, std::make_unique<out_t>(local_partitions[im]));
                        }
                    }
                }

                for (size_t im = 0; im < num_linear_models; ++im) {
                    auto j = frontier[im].second;
                    auto bpc = pfa::algorithm::epsilon_to_bpc(m[im].epsilon);
                    auto wkj = (j - k) * bpc + (sizeof(T1) + sizeof(T2)) * 8; //weight_ik(m[im], k, j, lossy);

                    if (distance[j] > distance[k] + wkj) {
                        distance[j] = distance[k] + wkj;
                        previous[j] = std::make_pair(bpc, std::make_unique<out_t>(local_partitions[im].copy(k)));
                    }

                }
            }

            //auto k = std::visit([](auto &&mo) { return mo.get_start(); }, local_partitions[num_models - 1]);
            auto k = n;
            while (k != 0) {
                auto bpc = previous[k].first;
                auto &f = previous[k].second;
                mem_out.emplace_back(bpc, *f);
                auto kp = f->get_start(); //std::visit([](auto &&mo) -> x_t { return mo.get_start(); }, *f);
                residuals_bit_size += (k - kp) * bpc;
                k = kp;
            }

            std::reverse(mem_out.begin(), mem_out.end());
            make_residuals(begin);
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

            auto l = bits_per_correction.size();
            auto it_end = starting_positions_ef.at(0);

            for (auto index_model_fun = 0; index_model_fun < l; ++index_model_fun) {
                auto end =
                        index_model_fun == (l - 1) ? n : *(++it_end);//starting_positions_select(index_model_fun + 2);

                //start = starting_positions[index_model_fun];
                bpc = bits_per_correction[index_model_fun];
                auto imt = index_model_fun;
                //auto mt = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];

                offset_coefficients++;
                auto model = typename pla_t::fun_t{start, t1, t2};
                for (auto j = start; j < end; ++j) {
                    uint64_t residual = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u),
                                                             offset_res & 0x3F, bpc);
                    offset_res += bpc;
                    auto y = model(j + 1);
                    auto _y = y + residual;
                    if (bpc != 0) _y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);

                    *(out_begin + j) = _y;
                }

                start = end;
            }
        }

        template<typename T = int64_t>
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

            auto unpack_pla = [this](x_t offset_coefficients, const auto num_residuals, auto *out_start) {
                //auto* xv = static_cast<float_scalar_t *>(std::aligned_alloc(64, _simd_width_bit_size));
                //std::iota(xv, xv + _simd_width, 1);

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];

                floatv_simd_t t1v, t2v;
                const floatv_simd_t startv([](int i) { return i + 1; });

                intv_simd_t _residuals{};

                auto j{0};
                for (; j + simd_width <= num_residuals; j += simd_width) {
                    t1v = floatv_simd_t{t1};
                    t2v = floatv_simd_t{t2};

                    //startv = stdx::ceil((startv + j) * t1v + t2v);
                    _residuals.copy_from(out_start + j, stdx::element_aligned);
                    _residuals += stdx::static_simd_cast<intv_simd_t>(stdx::ceil((startv + j) * t1v + t2v));
                    _residuals.copy_to(out_start + j, stdx::element_aligned);
                }

                for (; j < num_residuals; ++j) {
                    int_scalar_t _y = std::ceil(t1 * (j + 1) + t2);
                    *(out_start + j) += _y;
                }
            };

            uint8_t bpc{};
            x_t offset_res{0};
            auto it_end = starting_positions_ef.at(0);
            x_t offset_coefficients{0};

            x_t start{0};
            x_t end;

            constexpr auto np = 8;
            auto i_model{0};
            for (; i_model + np < bits_per_correction.size(); i_model += np) {

                uint8_t _bpc;
#pragma unroll
                for (std::size_t j{0}; j < np; ++j) {
                    end = *(++it_end);
                    _bpc = bits_per_correction[i_model + j];
                    if (_bpc != 0) unpack_residuals(i_model + j, offset_res, end - start, out + start);
                    unpack_pla(offset_coefficients + j, end - start, out + start);

                    offset_res += _bpc * (end - start);
                    start = end;
                }
                offset_coefficients += np;
            }

            for (; i_model < bits_per_correction.size(); ++i_model) {
                end = i_model == (bits_per_correction.size() - 1) ? _n : *(++it_end);
                bpc = bits_per_correction[i_model];
                if (bpc != 0) unpack_residuals(i_model, offset_res, end - start, out + start);
                unpack_pla(offset_coefficients, end - start, out + start);
                offset_coefficients++;
                offset_res += bpc * (end - start);
                start = end;
            }
        }

        size_t size_in_bits() const {
            return sizeof(*this) * 8 + residuals.bit_size() +  // offset_residuals.size() * sizeof(uint32_t) * 8 +
                   //sdsl::size_in_bytes(offset_residuals_ef) * 8 +
                   starting_positions_ef.size_in_bytes() * 8 +
                   //sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                   sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() +
                   // model_types_0.bit_size() + model_types_1.bit_size() + qbv.bit_size() +
                   offset_residuals_ef.size_in_bytes() * 8 + starting_positions_ef.size_in_bytes() * 8 +
                   //(sdsl::size_in_bytes(offset_residuals_ef_sls) + sdsl::size_in_bytes(starting_positions_select) +
                   // sdsl::size_in_bytes(starting_positions_rank) +
                   //(sdsl::size_in_bytes(fun_1_rank) + sdsl::size_in_bytes(quad_fun_rank)) * 8 +
                   //sdsl::size_in_bytes(starting_positions_ef) * 8 +
                   bits_per_correction.bit_size();
        }

        constexpr inline y_t operator[](x_t i) const {
            auto res = starting_positions_ef.predecessor(i);
            auto index_model = res.index();
            //auto index_model = it_model.index();
            //auto start_pos = static_cast<x_t>(starting_positions_select(index_model + 1));
            //auto start_pos = *it_model;
            uint64_t start_pos = *res;

            //auto imt = index_model;
            auto bpc = bits_per_correction[index_model];
            auto offset_residual = index_model == 0 ? 0 : offset_residuals_ef[index_model - 1];

            auto t1 = coefficients_t1[index_model];
            auto t2 = coefficients_t2[index_model];

            auto model = typename pla_t::fun_t(start_pos, t1, t2);
            const auto idx = offset_residual + bpc * (i - start_pos);

            auto residual = sdsl::bits::read_int(residuals.data() + (idx >> 6u), idx & 0x3F, bpc);
            auto _y = model(i + 1);
            auto y = _y + residual;
            if (bpc != 0) y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);
            return y;
        }

        /*
        size_t storage_size_in_bits() const {
            auto num_partitions = bits_per_correction.size();
            return residuals.bit_size() + sizeof(x_t) * 8 * coefficients_s.size() +
                   sizeof(T1) * 8 * coefficients_t0.size() +
                   sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() +
                   model_types_0.bit_size() + model_types_1.bit_size() + (num_partitions * sizeof(x_t) * 8) +
                   bits_per_correction.bit_size();
        }
        */

        size_t coefficients_size_in_bits() const {
            return sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size();
        }

        size_t residuals_size_in_bits() const {
            return residuals.bit_size();
        }

        void size_info(bool header = true) const {
            if (header) {
                std::cout << "residuals,offset_residuals,coefficients,starting_positions,bits_per_correction,meta\n";
            }
            std::cout << residuals.bit_size() << ",";
            std::cout << offset_residuals_ef.size_in_bytes() * 8 << ",";
            std::cout << coefficients_size_in_bits() << ",";
            std::cout << starting_positions_ef.size_in_bytes() * 8 << ",";
            std::cout << bits_per_correction.bit_size() << ",";
            std::cout << sizeof(*this) * 8 << std::endl;
        }

        /*
        void storage_size_info() const {
            auto num_partitions = model_types_0.size();
            std::cout << residuals.bit_size() << ","
                      << sizeof(x_t) * 8 * coefficients_s.size() + sizeof(T1) * 8 * coefficients_t0.size() +
                         sizeof(T1) * 8 * coefficients_t1.size() + sizeof(T2) * 8 * coefficients_t2.size() << ","
                      << model_types_0.bit_size() + model_types_1.bit_size() << ","
                      << num_partitions * sizeof(x_t) * 8 << ","
                      << bits_per_correction.bit_size() << ",";
        }
        */


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

        /*
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
        */

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

        /*
        void inline write_info_csv(std::ostream &ostream) {
            ostream.precision(16);
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
                auto end = index_model_fun == (l - 1) ? _n : *(++it_end);//starting_positions_select(index_model_fun + 2);
                ostream << index_model_fun << ",";

                //start = starting_positions[index_model_fun];
                bpc = bits_per_correction[index_model_fun];
                ostream << (uint64_t)(bpc) << ",";
                auto imt = index_model_fun;
                auto mt = (uint8_t) (model_types_0[imt]) | ((uint8_t) (model_types_1[imt]) << 1);
                ostream << (uint64_t)(mt) << ",";

                auto t1 = coefficients_t1[offset_coefficients];
                auto t2 = coefficients_t2[offset_coefficients];
                offset_coefficients++;
                std::optional<x_t> s = std::nullopt;
                std::optional<T1> t0 = std::nullopt;

                if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Sqrt) { // Too arbitrary?
                    s = coefficients_s[offset_coefficients_s++];
                    ostream << (uint64_t)(s.value()) << ",";
                } else if ((typename poa_t::approx_fun_t) (mt) == poa_t::approx_fun_t::Quadratic) {
                    t0 = coefficients_t0[offset_coefficients_t0++];
                    ostream << "," << (std::float64_t)(t0.value());
                } else {
                    ostream << ",";
                }
                ostream << "," << (std::float64_t)(t1) << "," << (std::float64_t)(t2) << ",";
                ostream << (end - start) << ",";
                auto model = poa_t::piecewise_non_linear_approximation::make_fun((typename poa_t::approx_fun_t) (mt),
                                                                                 start, s, t0, t1, t2);
                std::stringstream residual_str;
                for (auto j = start; j < end; ++j) {
                    uint64_t residual = sdsl::bits::read_int(residuals.data() + (offset_res >> 6u),
                                                             offset_res & 0x3F, bpc);
                    if (j == start)  residual_str << "[";
                    residual_str << residual;
                    if (j == (end - 1)) residual_str << "]";
                    else residual_str << ":";
                    offset_res += bpc;
                }
                ostream << residual_str.str() << std::endl;
                start = end;
            }
        }
        */

        /*
        static auto load(std::istream &is) {
            decltype(max_bpc) _max_bpc = 0;
            sdsl::read_member(_max_bpc, is);
            compressor<x_t, y_t, poly_t, T1, T2> lc{_max_bpc};
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
        */


    };
}