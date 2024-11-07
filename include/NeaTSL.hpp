#pragma once

#include "float_pfa.hpp"
#include <array>
#include <tuple>

namespace neats {

    template<typename x_t = uint32_t, typename y_t = int64_t, int64_t max_error = 8, typename poly = double, typename T1 = float, typename T2 = double>
    class lossy_compressor {
        using poa_t = pfa::piecewise_optimal_approximation<x_t, y_t, poly, T1, T2>;
        using polygon_t = poa_t::convex_polygon_t;
        //using fun_t = poa_t::fun_t;

        constexpr static auto m = typename poa_t::template pfa_t<max_error, typename poa_t::pla_t, typename poa_t::pea_t, typename poa_t::pqa_t, typename poa_t::psa_t>();

        constexpr static auto num_models = m.size;
        static_assert(num_models == 4, "The number of models must be 4");
        using out_t = typename decltype(m)::out_t;

        using approx_fun_t = typename poa_t::approx_fun_t;

        std::vector<out_t> out{};
        x_t _n = 0;

        std::vector<x_t> starting_positions;

        sdsl::int_vector<2> model_types{};

        std::vector<T1> coefficients_t0;
        std::vector<T1> coefficients_t1;
        std::vector<T2> coefficients_t2;
        std::vector<x_t> coefficients_s;

    public:

        inline auto num_partitions() const {
            return starting_positions.size();
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

            std::vector<std::unique_ptr<out_t>> previous(n + 1);

            polygon_t g{};
            distance[0] = 0;

            for (auto k = 0; k < n; ++k) {
                m.template for_each([&](auto &&model, auto &&imo) {
                    auto im = imo;

                    if (frontier[im].second <= k) { // an edge overlaps the current point (i.e. k)
                        auto t = pfa::algorithm::make_segment<poa_t>(model, g, (begin + k), end,
                                                                     frontier[im].second);
                        frontier[im].first = std::get<0>(t);
                        frontier[im].second = std::get<1>(t);
                        //assert(frontier[im].first < frontier[im].second - 1);
                        local_partitions[im] = std::get<2>(t);
                    }

                    if (frontier[im].second > k) {// relax prefix edge (i, k)
                        auto i = frontier[im].first;
                        //auto wik = (double) poa_t::size_in_bits() / ((k - i)*8);
                        auto wik = std::decay_t<decltype(model)>::fun_t::size_in_bits();

                        if (distance[k] > distance[i] + wik) {
                            distance[k] = distance[i] + wik;
                            //previous[k] = std::make_unique<out_t>(local_partitions[im]);
                            std::visit([&](auto &&p) {
                                previous[k] = std::make_unique<out_t>(p.copy(i));
                            }, local_partitions[im]);
                        }
                    }
                });

                //relax suffix edge (k, j)
                m.template for_each([&](auto &&model, auto &&imo) {
                    auto im = imo;
                    auto j = frontier[im].second;
                    auto wkj = std::decay_t<decltype(model)>::fun_t::size_in_bits();
                    if (distance[j] > distance[k] + wkj) {
                        distance[j] = distance[k] + wkj;
                        std::visit([&](auto &&p) {
                            previous[j] = std::make_unique<out_t>(p.copy(k));
                        }, local_partitions[im]);
                    }
                });
            }

            //auto k = std::visit([](auto &&mo) { return mo.get_start(); }, local_partitions[num_models - 1]);
            auto k = n;
            while (k != 0) {
                auto model = std::move(previous[k]);
                k = std::visit([](auto &&mo) { return mo.get_start(); }, *model);
                out.push_back(std::move(*model));
            }

            std::reverse(out.begin(), out.end());

            auto num_partitions = out.size();
            starting_positions = std::vector<x_t>(num_partitions, 0);
            model_types = sdsl::int_vector<2>(num_partitions, 0);

            for (auto index_model = 0; index_model < out.size(); ++index_model) {
                auto model = out[index_model];

                auto mt = std::visit([](auto &&mo) { return (uint8_t) mo.type(); }, model);
                model_types[index_model] = mt;
                starting_positions[index_model] = std::visit([](auto &&mo) {
                    return mo.get_start();
                }, model);

                std::visit([&](auto &&mo) {
                    auto t = mo.parameters();
                    if (std::get<0>(t).has_value()) {
                        coefficients_s.push_back(std::get<0>(t).value());
                    }

                    if (std::get<1>(t).has_value()) {
                        coefficients_t0.push_back(std::get<1>(t).value());
                    }
                    coefficients_t1.push_back(std::get<2>(t));
                    coefficients_t2.push_back(std::get<3>(t));
                }, model);
            }

            out.clear();
            out.shrink_to_fit();
        }

        template<typename It>
        inline void decompress(It out_begin, It out_end) const {

            const auto make_fun = [&](approx_fun_t mt, x_t start_pos, std::optional<x_t> d, std::optional<T1> t0, T1 t1,
                                      T2 t2) -> out_t {
                switch (mt) {
                    case approx_fun_t::Linear:
                        assert(!d.has_value());
                        return typename poa_t::pla_t::fun_t{start_pos, t1, t2};
                    case approx_fun_t::Quadratic:
                        assert(!d.has_value());
                        assert(t0.has_value());
                        return typename poa_t::pqa_t::fun_t{start_pos, t0.value(), t1, t2};
                    case approx_fun_t::Sqrt:
                        assert(d.has_value());
                        return typename poa_t::psa_t::fun_t{start_pos, d.value(), t1, t2};
                    case approx_fun_t::Exponential:
                        assert(!d.has_value());
                        return typename poa_t::pea_t::fun_t{start_pos, t1, t2};
                    default:
                        throw std::runtime_error("Not implemented");
                }
            };

            auto n = std::distance(out_begin, out_end);

            assert(n == _n);

            x_t start = starting_positions[0];

            auto offset_coefficients = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients_t0 = 0;

            auto l = starting_positions.size();
            for (auto index_model = 0; index_model < l; ++index_model) {

                auto end = index_model == (l - 1) ? n : starting_positions[index_model + 1];

                auto imt = index_model;
                auto mt = model_types[imt];

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

                auto model = make_fun((typename poa_t::approx_fun_t) (mt), start, s, t0, t1, t2);
                for (auto j = start; j < end; ++j) {
                    auto y = std::visit([&](auto &&mo) { return mo(j + 1); }, model);

                    *(out_begin + j) = y;
                }
                start = end;
            }
        }

        size_t size_in_bits() const {
            size_t size = 0;
            size += model_types.size() * 2;
            size += coefficients_t0.size() * sizeof(T1) * 8;
            size += coefficients_t1.size() * sizeof(T1) * 8;
            size += coefficients_t2.size() * sizeof(T2) * 8;
            size += coefficients_s.size() * sizeof(x_t) * 8;
            size += starting_positions.size() * sizeof(x_t) * 8;
            return size;
        }

        constexpr auto size() const {
            return _n;
        }

        constexpr static auto max_err() {
            return max_error;
        }

        std::vector<y_t> decompress() const {
            std::vector<y_t> res(_n);

            auto num_partitions = starting_positions.size();

            auto offset_coefficients_t0 = 0;
            auto offset_coefficients_s = 0;
            auto offset_coefficients = 0;

            uint32_t i_segment = 0;
            uint32_t start_segment = 0;
            uint32_t end_segment;
            x_t i = 0;

            while (i < _n) {
                if (i_segment >= (num_partitions - 1)) end_segment = _n;
                else end_segment = starting_positions[i_segment + 1];

                auto t = static_cast<poa_t::approx_fun_t>(model_types[i_segment]);
                switch (t) {
                    case poa_t::approx_fun_t::Linear: {
                        auto slope = coefficients_t1[offset_coefficients];
                        auto intercept = coefficients_t2[offset_coefficients];
                        offset_coefficients++;
                        auto start_pos = starting_positions[i_segment];
                        for (uint32_t j = start_segment; j < end_segment; ++j) {
                            res[i++] = std::ceil(slope * (j + 1 - start_pos) + intercept);
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Quadratic: {
                        const auto a = coefficients_t0[offset_coefficients_t0++];
                        const auto b = coefficients_t1[offset_coefficients];
                        const auto c = coefficients_t2[offset_coefficients];
                        offset_coefficients++;
                        auto start_pos = starting_positions[i_segment];
                        for (uint32_t j = start_segment; j < end_segment; ++j) {
                            res[i++] = std::ceil(a * (j - start_pos) * (j - start_pos) + b * (j - start_pos) + c);
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Exponential: {
                        const auto a = coefficients_t1[offset_coefficients];
                        const auto b = coefficients_t2[offset_coefficients];
                        offset_coefficients++;
                        auto start_pos = starting_positions[i_segment];
                        for (uint32_t j = start_segment; j < end_segment; ++j) {
                            res[i++] = std::ceil(std::exp(a * ((j - start_pos) + 1)) * b);
                        }
                        break;
                    }

                    case poa_t::approx_fun_t::Sqrt: {
                        const auto s = coefficients_s[offset_coefficients_s++];
                        const auto a = coefficients_t1[offset_coefficients];
                        const auto b = coefficients_t2[offset_coefficients];
                        offset_coefficients++;
                        auto start_pos = starting_positions[i_segment];
                        for (uint32_t j = start_segment; j < end_segment; ++j) {
                            res[i++] = std::ceil(a * std::sqrt((j - (start_pos - s) + 1)) + b);
                        }
                        break;
                    }
                }
                ++i_segment;
                start_segment = end_segment;
            }
            return res;
        }

    };
}