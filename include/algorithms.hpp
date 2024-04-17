#pragma once

#include "float_pfa.hpp"
#include <tuple>
#include <type_traits>
#include <cstdint>
#include <sdsl/int_vector.hpp>
#include <sux/bits/SimpleSelect.hpp>
#include <sux/bits/SimpleSelectHalf.hpp>
#include <sux/bits/SimpleSelectZero.hpp>
#include <sux/bits/SimpleSelectZeroHalf.hpp>
/** Computes (bits_per_correction > 0 ? 2^(bits_per_correction-1) - 1 : 0) without the conditional operator. */
#define BPC_TO_EPSILON(bits_per_correction) (((1ul << (bits_per_correction)) + 1) / 2 - 1)
#define LOG2(x) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll(x) - 1))

/** Computes the smallest integral value not less than x / y, where x and y must be positive integers. */
#define CEIL_UINT_DIV(x, y) ((x) / (y) + ((x) % (y) != 0))

namespace fa::algorithm {

    template<typename poa_t, typename It, typename T, bool quadratic = std::is_same_v<T, typename poa_t::pqa_t>>
    inline auto make_segment(const T &pa, typename poa_t::convex_polygon_t &g, It begin, It end, uint32_t start_x) {
        //using out_t = std::tuple<typename poa_t::x_t, typename poa_t::x_t, typename T::fun_t>;

        uint32_t n = std::distance(begin, end);
        typename poa_t::data_point last_starting_point;
        typename poa_t::data_point p0;

        bool intersect;
        for (uint32_t i = 1; i <= n; ++i) {
            //typename poa_t::y_t last_value;
            auto last_value = *(begin + (i - 1));
            auto dp = typename poa_t::data_point{i, last_value};

            if constexpr (quadratic) {
                if (i == 1) {
                    p0 = typename poa_t::data_point{i - 1, last_value};
                    last_starting_point = typename poa_t::data_point{start_x, last_value};
                    continue;
                } else {
                    dp.first = i - 1;
                    intersect = pa.add_point(g, p0, dp);
                }
            } else {
                intersect = pa.add_point(g, dp);
            }

            if (!intersect) {
                if constexpr (quadratic) {
                    auto f = pa.create_fun(g, last_starting_point);
                    g.clear();
                    return std::make_tuple(start_x, start_x + (i - 1), f);
                } else {
                    auto f = pa.create_fun(g, typename poa_t::data_point{start_x, dp.second});
                    g.clear();
                    return std::make_tuple(start_x, start_x + (i - 1), f);
                }
            }
        }

        if (!g.empty()) {
            if (quadratic) {
                auto f = pa.create_fun(g, last_starting_point);
                g.clear();
                return std::make_tuple(start_x, start_x + n, f);
            } else {
                auto f = pa.create_fun(g, typename poa_t::data_point{start_x, *(begin + (n - 1))});
                g.clear();
                return std::make_tuple(start_x, start_x + n, f);
            }
        } else {
            if constexpr (quadratic) {
                auto f = pa.create_fun(g, last_starting_point);
                g.clear();
                return std::make_tuple(start_x, start_x + n, f);
            }
            throw std::runtime_error("You should be not here");
        }
    }

    int64_t epsilon_to_bpc(int64_t epsilon) {
        if (epsilon == 0) return 0;
        else return (std::log2(epsilon + 1) + 1);
    }

    template<typename A, int64_t epsilon, typename It>
    auto make_pla(const It &begin, const It &end) {
        auto n = std::distance(begin, end);
        auto pa = A{epsilon};
        auto res = pa.make_approximation(begin, end);
        auto approx = pa.get_approximations(res, n);

        int64_t num_errors = 0;
        int64_t max_error = 0;
        for (auto i = 0; i < n; ++i) {
            auto err = static_cast<int64_t>(*(begin + i) - approx[i]);
            if ((err > epsilon) || (err < -(epsilon + 1))) {
                ++num_errors;
                auto abs_err = std::abs(err - epsilon);
                max_error = abs_err > max_error ? abs_err : max_error;
                std::cout << i << ": " << *(begin + i) << "!=" << approx[i] << std::endl;
            }
        }

        return res;
    }

}

