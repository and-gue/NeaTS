#pragma once

#include "float_pfa.hpp"
#include <array>
#include <tuple>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/util.hpp>
#include "my_elias_fano.hpp"
#include "vectorclass.h"

namespace neats {

    template<typename x_t = uint32_t, typename y_t = int64_t, int64_t max_error = 17L, typename poly = double, typename T1 = float, typename T2 = double, bool only_linear = false>
    class lossless_compressor {
        using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t, poly, T1, T2>;
        using polygon_t = poa_t::convex_polygon_t;
        //using fun_t = poa_t::fun_t;

        //constexpr static auto m = typename poa_t::template models<max_error, typename poa_t::pla_t, typename poa_t::pea_t, typename poa_t::psa_t, typename poa_t::ppa_t>();
        using model_bound_t = std::conditional_t<only_linear, typename poa_t::template la_vector<max_error>, typename poa_t::template models_t<max_error>>;
        constexpr static auto m = typename model_bound_t::tm{};
        constexpr static auto num_models = m.total_size();
        using out_t = typename decltype(m)::out_t;

        std::vector<std::pair<uint8_t, out_t>> mem_out{};
        x_t _n = 0;

        x_t residuals_bit_size = 0;

        //std::vector<x_t> starting_positions;
        MyEliasFano<true> starting_positions_ef;
        sdsl::int_vector<64> residuals;
        //std::vector<uint32_t> offset_residuals;
        //sdsl::sd_vector<> offset_residuals_ef;
        MyEliasFano<false> offset_residuals_ef;
        sdsl::int_vector<> bits_per_correction;
        //sdsl::int_vector<2> model_types;
        sdsl::bit_vector model_types_0;
        sdsl::bit_vector model_types_1;
        sdsl::bit_vector qbv;

        std::vector<T1> coefficients_t0;
        std::vector<T1> coefficients_t1;
        std::vector<T2> coefficients_t2;
        std::vector<x_t> coefficients_s;

        //sdsl::rank_support_v<1> starting_positions_rank;
        //sdsl::select_support_mcl<1> starting_positions_select;
        //sdsl::rank_support_sd<1> starting_positions_rank;
        //sdsl::select_support_sd<1> starting_positions_select;

        // each pattern depends on approx_fun_t enum
        sdsl::rank_support_v<1> fun_1_rank;
        sdsl::rank_support_v<1> quad_fun_rank;

        //sdsl::select_support_sd<1> offset_residuals_ef_sls;

    public:

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
                auto mt = std::visit([&](auto &&mo) { return (uint8_t) (mo.type()); }, model);

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
            mem_out.shrink_to_fit();
        }

        // from begin to end the data is already "normalized" (i.e. > 0)
        template<typename It>
        inline void partitioning(It begin, It end) {
            const auto n = std::distance(begin, end);

            _n = n;
            std::vector<int64_t> distance(n + 1, std::numeric_limits<int64_t>::max());
            std::array<std::pair<std::make_signed_t<x_t>, std::make_signed_t<x_t>>, num_models> frontier{
                    std::make_pair(0, 0)};
            std::array<out_t, num_models> local_partitions;

            std::vector<std::pair<uint8_t, std::unique_ptr<out_t>>> previous(n + 1);

            polygon_t g{};
            distance[0] = 0;

            for (auto k = 0; k < n; ++k) {

                m.template for_each([&](auto &&t, auto &&it) {
                    t.template for_each([&](auto &&model, auto &&imo) {
                        auto im = it * t.size + imo;

                        if (frontier[im].second <= k) { // an edge overlaps the current point (i.e. k)
                            auto t = fa::algorithm::make_segment<poa_t>(model, g, (begin + k), end,
                                                                        frontier[im].second);
                            frontier[im].first = std::get<0>(t);
                            frontier[im].second = std::get<1>(t);
                            //assert(frontier[im].first < frontier[im].second - 1);
                            local_partitions[im] = std::get<2>(t);
                        }

                        if (frontier[im].second > k) {// relax prefix edge (i, k)
                            auto i = frontier[im].first;
                            //auto wik = (double) poa_t::size_in_bits() / ((k - i)*8);
                            auto bpc = fa::algorithm::epsilon_to_bpc(model.epsilon);
                            auto wik = std::decay_t<decltype(model)>::fun_t::size_in_bits() +
                                       bpc * (k - i) + LOG2(_n);

                            if (distance[k] > distance[i] + wik) {
                                distance[k] = distance[i] + wik;
                                //previous[k] = std::make_unique<out_t>(local_partitions[im]);
                                std::visit([&](auto &&p) {
                                    previous[k] = std::make_pair(bpc, std::make_unique<out_t>(p.copy(i)));
                                }, local_partitions[im]);
                            }
                        }
                    });
                });

                //relax suffix edge (k, j)
                m.template for_each([&](auto &&t, auto &&it) {
                    t.template for_each([&](auto &&model, auto &&imo) {
                        auto im = it * t.size + imo;
                        auto j = frontier[im].second;
                        auto bpc = fa::algorithm::epsilon_to_bpc(model.epsilon);
                        auto wkj = std::decay_t<decltype(model)>::fun_t::size_in_bits() +
                                   bpc * (j - k) + LOG2(_n);
                        if (distance[j] > distance[k] + wkj) {
                            distance[j] = distance[k] + wkj;
                            std::visit([&](auto &&p) {
                                previous[j] = std::make_pair(bpc, std::make_unique<out_t>(p.copy(k)));
                            }, local_partitions[im]);
                        }
                    });
                });
            }

            //auto k = std::visit([](auto &&mo) { return mo.get_start(); }, local_partitions[num_models - 1]);
            auto k = n;
            while (k != 0) {
                auto bpc = previous[k].first;
                auto model = std::move(previous[k].second);
                mem_out.emplace_back(bpc, *model);
                auto kp = std::visit([](auto &&mo) { return mo.get_start(); }, *model);
                residuals_bit_size += (k - kp) * bpc;
                k = kp;
            }

            std::reverse(mem_out.begin(), mem_out.end());
            make_residuals(begin, end);
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

                auto model = model_bound_t::make_fun((typename poa_t::approx_fun_t) (mt), start, s, t0, t1, t2);
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

        void size_info() const {
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

            auto model = model_bound_t::make_fun((typename poa_t::approx_fun_t) (type_model), start_pos, s, t0, t1, t2);
            const auto idx = offset_residual + bpc * (i - start_pos);

            auto residual = sdsl::bits::read_int(residuals.data() + (idx >> 6u), idx & 0x3F, bpc);
            auto _y = std::visit([&](auto &&mo) { return mo(i + 1); }, model);
            auto y = _y + residual;
            if (bpc != 0) y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);
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

        constexpr static auto max_err() {
            return max_error;
        }

        inline size_t serialize(std::ostream &os, sdsl::structure_tree_node *v = nullptr,
                                const std::string &name = "") const {
            if (_n == 0) {
                throw std::runtime_error("compressor empty");
            }

            auto child = sdsl::structure_tree::add_child(v, name, sdsl::util::class_name(*this));
            size_t written_bytes = 0;
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

        static auto load(std::istream &is) {
            lossless_compressor<x_t, y_t, max_error, poly, T1, T2> lc;
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

        void decompress_SIMD(y_t *out) const {
            //using VecType = Vec8d;
            constexpr int64_t vec_size = 8;

            const auto n = _n;
            const uint64_t *residuals_ptr = residuals.data();

            uint32_t start_segment = 0;
            auto it_end_segment = starting_positions_ef.at(0);
            uint32_t end_segment = *it_end_segment;

            uint8_t bpc;
            uint32_t offset_res = 0;
            auto offset_coefficients = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients_t0 = 0;

            const auto nmo = bits_per_correction.size();
            x_t i = 0;

            for (uint32_t i_segment = 0; i_segment < nmo; ++i_segment) {
                end_segment = i_segment == (nmo - 1) ? n : *(++it_end_segment);
                bpc = bits_per_correction[i_segment];
                int64_t epsilon = BPC_TO_EPSILON(bpc) + 1;

                const auto t1 = coefficients_t1[offset_coefficients];
                const auto t2 = coefficients_t2[offset_coefficients];
                offset_coefficients++;

                auto mt = static_cast<poa_t::approx_fun_t>((uint8_t) (model_types_0[i_segment]) |
                                                           ((uint8_t) (model_types_1[i_segment]) << 1));
                auto len_segment = end_segment - start_segment;

                switch (mt) {
                    case poa_t::approx_fun_t::Linear: {
                        Vec8d i_vec(1, 2, 3, 4, 5, 6, 7, 8);
                        Vec8q residuals_vec;

                        int64_t j = 0;
                        for (; j + vec_size <= len_segment; j += vec_size) {
                            // h index inside the simd word
#pragma GCC ivdep
                            for (int64_t h = 0; h < vec_size; ++h) {
                                uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                         offset_res & 0x3F, bpc);
                                offset_res += bpc;
                                residuals_vec.insert(h, residual);
                            }

                            Vec8q dst = _mm512_cvtpd_epi64(ceil(t1 * i_vec + t2));

                            dst += residuals_vec;
                            if (bpc != 0) dst -= epsilon;

                            dst.store(out + i);
                            i_vec += vec_size;
                            i += vec_size;
                        }

                        for (; j < len_segment; ++j) {
                            uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                     offset_res & 0x3F, bpc);
                            offset_res += bpc;
                            auto y = ceil(t1 * static_cast<double>(j + 1) + t2);
                            auto res = y + residual;
                            if (bpc != 0) res -= epsilon;
                            out[i++] = res;
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Quadratic: {
                        Vec8d i_vec(0, 1, 2, 3, 4, 5, 6, 7);
                        Vec8q residuals_vec;

                        auto t0 = coefficients_t0[offset_coefficients_t0++];
                        int64_t j = 0;
                        for (; j + vec_size <= len_segment; j += vec_size) {
                            // h index inside the simd word
#pragma GCC ivdep
                            for (int64_t h = 0; h < vec_size; ++h) {
                                uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                         offset_res & 0x3F, bpc);
                                offset_res += bpc;
                                residuals_vec.insert(h, (int64_t) residual);
                            }

                            Vec8q dst = _mm512_cvtpd_epi64(ceil((i_vec * i_vec * t0) + (i_vec * t1) + t2));
                            dst += residuals_vec;
                            if (bpc != 0) dst -= epsilon;

                            dst.store(out + i);
                            i_vec += vec_size;
                            i += vec_size;
                        }

                        for (; j < len_segment; ++j) {
                            uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                     offset_res & 0x3F, bpc);
                            offset_res += bpc;
                            auto x = static_cast<double>(j);
                            auto y = ceil(x * x * t0 + x * t1 + t2);
                            auto res = y + residual;
                            if (bpc != 0) res -= epsilon;
                            out[i++] = res;
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Sqrt: {
                        auto s = static_cast<double>(coefficients_s[offset_coefficients_s++]);
                        auto model = model_bound_t::make_fun(mt, start_segment, s, std::nullopt, t1, t2);
                        for (auto j = start_segment; j < end_segment; ++j) {
                            uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                     offset_res & 0x3F, bpc);
                            offset_res += bpc;
                            auto y = std::visit([&](auto &&mo) { return mo(j + 1); }, model);
                            auto _y = y + residual;
                            if (bpc != 0) _y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);
                            out[i++] = _y;
                        }
                        /*
                        Vec8d i_vec(1, 2, 3, 4, 5, 6, 7, 8);
                        auto s = static_cast<double>(coefficients_s[offset_coefficients_s++]);

                        Vec8d sqrt_i_vec = sqrt(i_vec + s);
                        Vec8q residuals_vec;

                        int64_t j = 0;
                        for (; j + vec_size <= len_segment; j += vec_size) {
                            // h index inside the simd word
#pragma GCC ivdep

                            for (int64_t h = 0; h < vec_size; ++h) {
                                uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                         offset_res & 0x3F, bpc);
                                offset_res += bpc;
                                residuals_vec.insert(h, (int64_t) residual);
                            }

                            Vec8q dst = _mm512_cvtpd_epi64(round((sqrt_i_vec * t1 + t2)));
                            dst += residuals_vec;
                            if (bpc != 0) dst -= epsilon;

                            dst.store(out + i);
                            i_vec += vec_size;
                            sqrt_i_vec = sqrt(i_vec + s);
                            i += vec_size;
                        }

                        for (; j < len_segment; ++j) {
                            uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                     offset_res & 0x3F, bpc);
                            offset_res += bpc;
                            auto x = static_cast<double>(j + 1 + s);
                            auto y = round(t1 * sqrt(x) + t2);
                            auto res = y + residual;
                            if (bpc != 0) res -= epsilon;
                            out[i++] = res;
                        }
                        */
                        break;
                    }

                    case poa_t::approx_fun_t::Exponential: {
                        auto model = model_bound_t::make_fun(mt, start_segment, std::nullopt, std::nullopt, t1, t2);
                        for (auto j = start_segment; j < end_segment; ++j) {
                            uint64_t residual = sdsl::bits::read_int(residuals_ptr + (offset_res >> 6u),
                                                                     offset_res & 0x3F, bpc);
                            offset_res += bpc;
                            auto y = std::visit([&](auto &&mo) { return mo(j + 1); }, model);
                            auto _y = y + residual;
                            if (bpc != 0) _y -= static_cast<y_t>(BPC_TO_EPSILON(bpc) + 1);
                            out[i++] = _y;
                        }
                        break;
                    }
                }

                start_segment = end_segment;
            }
        }
    };
}