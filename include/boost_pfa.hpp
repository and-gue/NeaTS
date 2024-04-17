#pragma once

#include "float_coefficient_space.hpp"
#include "algorithms.hpp"

namespace fa::pfa {

    template<typename X = uint32_t, typename Y = int64_t, typename polygon_t = double, typename T1 = float, typename T2 = double>
    struct piecewise_optimal_approximation {

        using x_t = X;
        using y_t = Y;

        using data_point = typename std::pair<x_t, y_t>;

        using convex_polygon_t = fa::fcs::convex_polygon<polygon_t>;
        using upperbound_t = typename convex_polygon_t::upperbound_t;
        using lowerbound_t = typename convex_polygon_t::lowerbound_t;
        using boundaries_t = typename convex_polygon_t::boundaries_t;
        using point_t = typename convex_polygon_t::point_t;

        enum class approx_fun_t {
            Linear = 0, Quadratic = 1, Sqrt = 2, Exponential = 3, COUNT = 4, Power = 5
        };

    private:

        struct piecewise_linear_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_linear_approximation() : epsilon{2L} {}

            constexpr explicit piecewise_linear_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point &p) const {
                assert(p.first > 0);

                auto m = -static_cast<y_t>(p.first);

                auto _uq = p.second + epsilon;
                auto _lq = p.second - epsilon;

                lowerbound_t l{convex_polygon_t::segment_t(m, _lq)};
                upperbound_t u{convex_polygon_t::segment_t(m, _uq)};
                return {u, l};
            }

            struct linear {

                x_t starting_position;
                T1 a;
                T2 b;

                constexpr linear() : starting_position{0}, a{}, b{} {}

                constexpr linear(x_t pos, const T1 &m, const T2 &q) : starting_position{pos}, a(m), b{q} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const long double x = i - starting_position;
                    return ceill(static_cast<long double>(a) * x + static_cast<long double>(b));
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + std::log2((uint64_t) approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                inline linear copy(x_t s1) const {
                    auto z = s1 - starting_position;
                    auto y = a * z + b;
                    return linear{s1, a, T2(y)};
                }

                inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(std::nullopt, std::nullopt, a, b);
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Linear;
                }
            };

            using fun_t = linear;

            inline bool add_point(convex_polygon_t &g, const data_point &p_start) const {
                bool intersect;
                auto [l, u] = compute_bounds(p_start);
                intersect = g.update(l, u);
                return intersect;
            }

            static constexpr linear create_fun(const convex_polygon_t &g, const data_point &p) {
                if (g.empty()) {
                    return linear{p.first, 0.0, static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._q + l0._q) / 2.0;
                    return linear{p.first, 0.0, static_cast<T2>(b)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    //std::cout <<"PL: " << pl.x() << ", " << pl.y() << std::endl;
                    //std::cout <<"PR: " << pr.x() << ", " << pr.y() << std::endl;

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    //std::cout <<"INTERCEPT: " << mp_a << " SLOPE: " << mp_b << std::endl;
                    //auto mp_a = pl.x();
                    //auto mp_b = pl.y();
                    return linear{p.first, static_cast<T1>(mp_a), static_cast<T2>(mp_b)};
                }
            }

            inline auto get_approximations(const std::vector<linear>& data, const uint32_t n) const {
                std::vector<y_t> result;

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result.emplace_back(f(i + 1));
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<linear> make_approximation(It begin, It end) const {

                convex_polygon_t g{};
                x_t n = std::distance(begin, end);
                std::vector<linear> result;
                result.reserve(n/10);

                x_t last = 0;
                for (x_t i = 1; i <= n; ++i) {
                    y_t last_value = *(begin + (i - 1));
                    auto dp = data_point{i - last, last_value};
                    auto [u, l] = compute_bounds(dp);
                    bool intersect = g.update(u, l);
                    if (!intersect) {
                        auto f = create_fun(g, data_point{last, last_value});
                        result.emplace_back(f);
                        g.clear();
                        last = --i;
                        /*
                        for (auto j = f.starting_position; j < last; ++j) {
                            auto approx = f(j + 1);
                            auto y = *(begin + j);
                            auto err = std::abs(approx - y);
                            if (err > (epsilon + 1)) {
                                std::cout << approx  << "!=" << y << std::endl;
                                std::cout << "INDEX: " << i << std::endl;
                                std::cout << "START: " << f.starting_position << std::endl;
                                std::cout << "END: " << last << std::endl;
                                std::cout << "ERROR" << std::endl;
                            }
                        }
                        */

                    }
                }

                if (!g.empty()) {
                    auto f = create_fun(g, data_point{last, *(begin + (n - 1))});
                    result.emplace_back(f);
                    g.clear();
                }

                return result;
            }

        };

        struct piecewise_power_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_power_approximation() : epsilon{2L} {}

            constexpr explicit piecewise_power_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point &p) const {
                assert(p.first > 0);

                auto m = -std::log(static_cast<convex_polygon_t::value_t>(p.first + 1));

                auto _uq = std::log(static_cast<convex_polygon_t::value_t>(p.second + epsilon));
                auto _lq = std::log(static_cast<convex_polygon_t::value_t>(p.second - epsilon))

                assert(_uq.min() >= _lq.max());
                lowerbound_t l{m, _lq};
                upperbound_t u{m, _uq};
                return {u, l};
            }

            struct power {

                x_t starting_position;
                x_t d = 0;
                T1 a;
                T2 b;

                constexpr power() : starting_position{0}, a{}, b{} {}

                constexpr power(x_t pos, const T1 &_a, const T2 &_b) : starting_position{pos}, a(_a), b{_b} {}
                constexpr power(x_t pos0, x_t pos1, const T1 &_a, const T2 &_b) : starting_position{pos0}, d{pos1}, a{_a}, b{_b} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const long double x = (i + 1) - (starting_position - d);
                    return ceill(static_cast<long double>(b) * pow(x, a));
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    return (sizeof(x_t) + sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + std::log2((uint64_t) approx_fun_t::COUNT);
                }

                inline power copy(x_t s1) const {
                    auto z = s1 - starting_position;
                    return power{s1, z, a, b};
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(d, std::nullopt, a, b);
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Power;
                }
            };

            using fun_t = power;

            inline bool add_point(convex_polygon_t &g, const data_point &p_start) const {
                bool intersect;
                auto [l, u] = compute_bounds(p_start);
                intersect = g.update(l, u);
                return intersect;
            }

            static constexpr power create_fun(const convex_polygon_t &g, const data_point &p) {
                if (g.empty()) {
                    return power{p.first, 0.0, T2(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._q + l0._q) / 2.0;
                    return power{p.first, 0.0, static_cast<T2>(std::exp(b))};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    //std::cout <<"PL: " << pl.x() << ", " << pl.y() << std::endl;
                    //std::cout <<"PR: " << pr.x() << ", " << pr.y() << std::endl;

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pl.y() + pr.y()) / 2.0;

                    //std::cout <<"INTERCEPT: " << mp_a << " SLOPE: " << mp_b << std::endl;
                    //auto mp_a = pl.x();
                    //auto mp_b = pl.y();
                    return power{p.first, static_cast<T1>(mp_a), static_cast<T2>(std::exp(mp_b))};
                }
            }

            inline auto get_approximations(const std::vector<power>& data, const uint32_t n) const {
                std::vector<y_t> result;

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result.emplace_back(f(i + 1));
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<power> make_approximation(It begin, It end) const {

                convex_polygon_t g{};
                x_t n = std::distance(begin, end);
                std::vector<power> result;
                result.reserve(n/10);

                x_t last = 0;
                for (x_t i = 1; i <= n; ++i) {
                    y_t last_value = *(begin + (i - 1));
                    auto dp = data_point{i - last, last_value};
                    auto [u, l] = compute_bounds(dp);
                    bool intersect = g.update(u, l);
                    if (!intersect) {
                        auto f = create_fun(g, data_point{last, last_value});
                        result.emplace_back(f);
                        g.clear();
                        last = --i;
                        /*
                        for (auto j = f.starting_position; j < last; ++j) {
                            auto approx = f(j + 1);
                            auto y = *(begin + j);
                            auto err = std::abs(approx - y);
                            if (err > (epsilon + 1)) {
                                std::cout << approx  << "!=" << y << std::endl;
                                std::cout << "INDEX: " << i << std::endl;
                                std::cout << "START: " << f.starting_position << std::endl;
                                std::cout << "END: " << last << std::endl;
                                std::cout << "ERROR" << std::endl;
                            }
                        }
                        */

                    }
                }

                if (!g.empty()) {
                    auto f = create_fun(g, data_point{last, *(begin + (n - 1))});
                    result.emplace_back(f);
                    g.clear();
                }

                return result;
            }

        };

        struct piecewise_sqrt_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_sqrt_approximation() : epsilon{2L} {}

            constexpr explicit piecewise_sqrt_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point &p) const {
                assert(p.first > 0);

                auto m = -std::sqrt(static_cast<convex_polygon_t::value_t>(p.first));

                auto _uq = p.second + epsilon;
                auto _lq = p.second - epsilon;

                lowerbound_t l{m, _lq};
                upperbound_t u{m, _uq};
                return {u, l};
            }

            struct sqrt {

                x_t starting_position;
                x_t d = 0;
                T1 a;
                T2 b;

                constexpr sqrt() : starting_position{0}, a{}, b{} {}

                constexpr sqrt(x_t pos, const T1 _a, const T2 _b) : starting_position{pos}, a{_a}, b{_b} {}
                constexpr sqrt(x_t pos0, x_t pos1, const T1 _a, const T2 _b) : starting_position{pos0}, d{pos1}, a{_a}, b{_b} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const long double x = i - (starting_position - d);
                    return ceill(static_cast<long double>(a) * std::sqrt(x) + static_cast<long double>(b));
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    return (sizeof(x_t) + sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + std::log2((uint64_t)approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                [[nodiscard]] inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(d, std::nullopt, a, b);
                }

                inline sqrt copy(x_t s1) const {
                    auto z = s1 - starting_position;
                    return sqrt{s1, z, a, b};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Sqrt;
                }
            };

            using fun_t = sqrt;

            inline bool add_point(convex_polygon_t &g, const data_point &p_start) const {
                bool intersect;
                auto [l, u] = compute_bounds(p_start);
                intersect = g.update(l, u);
                return intersect;
            }

            static constexpr sqrt create_fun(const convex_polygon_t &g, const data_point &p) {
                if (g.empty()) {
                    return sqrt{p.first, 0.0, static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._q + l0._q) / 2.0;
                    return sqrt{p.first, 0.0, static_cast<T2>(b)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    return sqrt{p.first, static_cast<T1>(mp_a), static_cast<T2>(mp_b)};
                }
            }

            inline auto get_approximations(const std::vector<sqrt>& data, const uint32_t n) const {
                std::vector<y_t> result;

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result.emplace_back(f(i + 1));
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<sqrt> make_approximation(It begin, It end) const {

                convex_polygon_t g{};
                x_t n = std::distance(begin, end);
                std::vector<sqrt> result;
                result.reserve(n/10);

                x_t last = 0;
                for (x_t i = 1; i <= n; ++i) {
                    y_t last_value = *(begin + (i - 1));
                    auto dp = data_point{i - last, last_value};
                    auto [u, l] = compute_bounds(dp);
                    bool intersect = g.update(u, l);
                    if (!intersect) {
                        auto f = create_fun(g, data_point{last, last_value});
                        result.emplace_back(f);
                        g.clear();
                        last = --i;
                        /*
                        for (auto j = f.starting_position; j < last; ++j) {
                            auto approx = f(j + 1);
                            auto y = *(begin + j);
                            auto err = std::abs(approx - y);
                            if (err > (epsilon + 1)) {
                                std::cout << approx  << "!=" << y << std::endl;
                                std::cout << "INDEX: " << i << std::endl;
                                std::cout << "START: " << f.starting_position << std::endl;
                                std::cout << "END: " << last << std::endl;
                                std::cout << "ERROR" << std::endl;
                            }
                        }
                        */

                    }
                }

                if (!g.empty()) {
                    auto f = create_fun(g, data_point{last, *(begin + (n - 1))});
                    result.emplace_back(f);
                    g.clear();
                }

                return result;
            }
        };

        struct piecewise_exponential_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_exponential_approximation() : epsilon{2L} {}

            constexpr explicit piecewise_exponential_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point &p) const {
                assert(p.first > 0);

                auto m = -static_cast<convex_polygon_t::value_t>(p.first);

                auto _uq = std::log(static_cast<convex_polygon_t::value_t>(p.second + epsilon));
                auto _lq = std::log(static_cast<convex_polygon_t::value_t>(p.second - epsilon));

                assert(_uq.min() >= _lq.max());
                lowerbound_t l{m, _lq};
                upperbound_t u{m, _uq};
                return {u, l};
            }

            struct exponential {

                x_t starting_position;
                T1 a;
                T2 b;

                constexpr exponential() : starting_position{0}, a{}, b{} {}

                constexpr exponential(x_t pos, const T1 _a, const T2 _b) : starting_position{pos}, a{_a}, b{_b} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const long double x = i - starting_position;
                    return ceill(b * std::exp(a * x));
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + std::log2((uint64_t)approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                [[nodiscard]] inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(std::nullopt, std::nullopt, a, b);
                }

                inline exponential copy(x_t x) const {
                    auto k = x - starting_position;
                    auto a1 = a;
                    auto b1 = b * std::exp(a * k);
                    return exponential{x, static_cast<T1>(a1), static_cast<T2>(b1)};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Exponential;
                }
            };

            using fun_t = exponential;

            inline bool add_point(convex_polygon_t &g, const data_point &p_start) const {
                bool intersect;
                auto [l, u] = compute_bounds(p_start);
                intersect = g.update(l, u);
                return intersect;
            }

            static constexpr exponential create_fun(const convex_polygon_t &g, const data_point &p) {
                if (g.empty()) {
                    return exponential{p.first, 0.0, static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._q + l0._q) / 2.0;
                    return exponential{p.first, 0.0, static_cast<T2>(std::exp(b))};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    return exponential{p.first, static_cast<T1>(mp_a), static_cast<T2>(std::exp(mp_b))};
                }
            }

            inline auto get_approximations(const std::vector<exponential>& data, const uint32_t n) const {
                std::vector<y_t> result;

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result.emplace_back(f(i + 1));
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<exponential> make_approximation(It begin, It end) const {

                convex_polygon_t g{};
                x_t n = std::distance(begin, end);
                std::vector<exponential> result;
                result.reserve(n/10);

                x_t last = 0;
                for (x_t i = 1; i <= n; ++i) {
                    y_t last_value = *(begin + (i - 1));
                    auto dp = data_point{i - last, last_value};
                    auto [u, l] = compute_bounds(dp);
                    bool intersect = g.update(u, l);
                    if (!intersect) {
                        auto f = create_fun(g, data_point{last, last_value});
                        result.emplace_back(f);
                        g.clear();
                        last = --i;
                        /*
                        for (auto j = f.starting_position; j < last; ++j) {
                            auto approx = f(j + 1);
                            auto y = *(begin + j);
                            auto err = std::abs(approx - y);
                            if (err > (epsilon + 1)) {
                                std::cout << approx  << "!=" << y << std::endl;
                                std::cout << "INDEX: " << i << std::endl;
                                std::cout << "START: " << f.starting_position << std::endl;
                                std::cout << "END: " << last << std::endl;
                                std::cout << "ERROR" << std::endl;
                            }
                        }
                        */

                    }
                }

                if (!g.empty()) {
                    auto f = create_fun(g, data_point{last, *(begin + (n - 1))});
                    result.emplace_back(f);
                    g.clear();
                }

                return result;
            }
        };

        struct piecewise_quadratic_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_quadratic_approximation() : epsilon{2L} {}

            constexpr explicit piecewise_quadratic_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point& p0, const data_point &p) const {

                auto m = -static_cast<convex_polygon_t::value_t>(p0.first + p.first);

                auto _lq = ((p.second - epsilon) - p0.second) / static_cast<convex_polygon_t::value_t>(p.first - p0.first);
                auto _uq = ((p.second + epsilon) - p0.second) / static_cast<convex_polygon_t::value_t>(p.first - p0.first);

                assert(_uq.min() >= _lq.max());
                lowerbound_t l{m, _lq};
                upperbound_t u{m, _uq};
                return {u, l};
            }

            struct quadratic {

                x_t starting_position;
                T1 a;
                T1 b;
                T2 c;

                constexpr quadratic() : starting_position{0}, a{}, b{}, c{} {}

                constexpr quadratic(x_t pos, const T1 _a, const T1 _b, const T2 _c) : starting_position{pos}, a{_a}, b{_b}, c{_c} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const long double x = (i - 1) - starting_position;
                    return ceill(a * x * x + b * x + c);
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T1) + sizeof(T2)) * 8 + std::log2((uint64_t)approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                [[nodiscard]] inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(std::nullopt, a, b, c);
                }

                /*
                inline void set_starting_position(x_t x) {
                    starting_position = x;
                }
                */

                inline quadratic copy(x_t x) const {
                    auto k = x - starting_position;
                    auto a1 = a;
                    auto b1 = 2 * a * k + b;
                    auto c1 = a * k * k + b * k + c;
                    return quadratic{x, static_cast<T1>(a1), static_cast<T1>(b1), static_cast<T2>(c1)};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Quadratic;
                }
            };

            using fun_t = quadratic;

            inline bool add_point(convex_polygon_t &g, const data_point&p0, const data_point &p) const {
                bool intersect;
                auto [l, u] = compute_bounds(p0, p);
                intersect = g.update(l, u);
                return intersect;
            }

            static constexpr quadratic create_fun(const convex_polygon_t &g, const data_point &p) {
                if (g.empty()) {
                    return quadratic{p.first, 0.0, 0.0, static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._q + l0._q) / 2.0;
                    return quadratic{p.first, 0.0, static_cast<T1>(b), static_cast<T2>(p.second)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    return quadratic{p.first, static_cast<T1>(mp_a), static_cast<T1>(mp_b), static_cast<T2>(p.second)};
                }
            }

            inline auto get_approximations(const std::vector<quadratic>& data, const uint32_t n) const {
                std::vector<y_t> result;

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result.emplace_back(f(i + 1));
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<quadratic> make_approximation(It begin, It end) {

                convex_polygon_t g{};
                x_t n = std::distance(begin, end);
                std::vector<quadratic> result;
                result.reserve(n/10);

                auto last_starting_point = data_point{0, *begin};
                auto p0 = data_point{0, *begin};
                for (x_t i = 1; i < n; ++i) {
                    y_t last_value = *(begin + i);
                    auto dp = data_point{(i - last_starting_point.first), last_value};
                    bool intersect = add_point(g, p0, dp);
                    if (!intersect) {
                        auto f = create_fun(g, last_starting_point);
                        result.emplace_back(f);
                        g.clear();
                        last_starting_point = data_point{i, last_value};
                        p0.second = last_value;

                        /*
                        for (auto j = f.starting_position; j < i; ++j) {
                            auto approx = f(j + 1);
                            auto y = *(begin + j);
                            auto err = static_cast<y_t>(y - approx);
                            if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
                                std::cout << approx  << "!=" << y << std::endl;
                                std::cout << "INDEX: " << i << std::endl;
                                std::cout << "START: " << f.starting_position << std::endl;
                                std::cout << "ERROR" << std::endl;

                                exit(1);
                            }
                        }
                        */

                    }
                }

                if (!g.empty()) {
                    auto f = create_fun(g, last_starting_point);
                    result.emplace_back(f);
                    g.clear();
                }

                return result;
            }
        };



    public:
        using pla_t = piecewise_linear_approximation;
        using ppa_t = piecewise_power_approximation;
        using psa_t = piecewise_sqrt_approximation;
        using pea_t = piecewise_exponential_approximation;
        using pqa_t = piecewise_quadratic_approximation;

        template<int64_t error, typename... Pfa> class pfa_t {
            std::tuple<Pfa...> pfa;

            template<typename First, typename ...Rest>
            constexpr inline void init(First &&first, Rest &&...rest) {
                std::get<size - sizeof...(Rest) - 1>(pfa) = std::forward<First>(first);
                if constexpr (sizeof...(Rest) > 0)
                    init(std::forward<Rest>(rest)...);
            }

            template<typename Tuple, typename F, std::size_t... Is>
            constexpr void for_each_impl(Tuple&& t, F&& f, std::index_sequence<Is...>) const {
                (f(std::get<Is>(t), Is), ...);
            }

        public:

            template<typename F>
            constexpr void for_each(F&& f) const {
                for_each_impl(pfa, f, std::make_index_sequence<size>{});
            }

            static constexpr size_t size = std::tuple_size_v<decltype(pfa)>;
            using out_t = std::variant<typename Pfa::fun_t...>;

            constexpr pfa_t() : pfa() {
                init(Pfa{error}...);
            }
        };

        template<uint64_t max_error, typename... Pfa> class models {

            template<std::size_t... Is>
            constexpr static auto make_models(std::index_sequence<Is...>) {
                return std::make_tuple(pfa_t<BPC_TO_EPSILON(Is + (Is >= 1)), Pfa...>{}...);
            }

            decltype(make_models(std::make_index_sequence<max_error>{})) m;

            template<typename Tuple, typename F, std::size_t... Is>
            constexpr void for_each_impl(Tuple&& t, F&& f, std::index_sequence<Is...>) const {
                (f(std::get<Is>(t), Is), ...);
            }

        public:

            constexpr models() : m() {
                m = make_models(std::make_index_sequence<max_error>{});
            }

            template<typename F>
            constexpr void for_each(F&& f) const {
                for_each_impl(m, f, std::make_index_sequence<max_error>{});
            }

            constexpr static auto total_size() {
                return sizeof...(Pfa) * max_error;
            }

            using out_t = typename pfa_t<max_error, Pfa...>::out_t;
        };

        template<uint64_t max_error, typename T = models<max_error, pla_t, pqa_t, psa_t, pea_t>>
        struct models_t {
            using tm = T;

            static inline auto make_fun(approx_fun_t mt, x_t start_pos, std::optional<x_t> d, std::optional<T1> t0, T1 t1, T2 t2) {
                switch (mt) {
                    case approx_fun_t::Linear:
                        assert(!d.has_value());
                        return typename tm::out_t{typename pla_t::fun_t{start_pos, t1, t2}};
                    case approx_fun_t::Quadratic:
                        assert(!d.has_value());
                        assert(t0.has_value());
                        return typename tm::out_t{typename pqa_t::fun_t{start_pos, t0.value(), t1, t2}};
                        /*
                    case approx_fun_t::Power:
                        assert(d.has_value());
                        return typename tm::out_t{typename ppa_t::fun_t{start_pos, d.value(), t1, t2}};
                         */
                    case approx_fun_t::Sqrt:
                        assert(d.has_value());
                        return typename tm::out_t{typename psa_t::fun_t{start_pos, d.value(), t1, t2}};
                    case approx_fun_t::Exponential:
                        assert(!d.has_value());
                        return typename tm::out_t{typename pea_t::fun_t{start_pos, t1, t2}};
                    default:
                        throw std::runtime_error("Not implemented");
                }
            }
        };

        template<uint64_t max_error, typename T = models<max_error, pla_t>>
        struct la_vector {
            using tm = T;


            static inline auto make_fun(approx_fun_t mt, x_t start_pos, std::optional<x_t> d, std::optional<T1> t0, T1 t1, T2 t2) {
                switch (mt) {
                    case approx_fun_t::Linear:
                        assert(!d.has_value());
                        return typename tm::out_t{typename pla_t::fun_t{start_pos, t1, t2}};
                    default:
                        throw std::runtime_error("Not implemented");
                }
            }
        };

    };
}