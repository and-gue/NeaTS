#pragma once

#include "float_dpfa.hpp"
#include "algorithms.hpp"

namespace pfa {

    template<typename X = uint32_t, typename Y = int64_t, typename polygon_t = double, typename T1 = float, typename T2 = double>
    struct piecewise_optimal_approximation {

        using x_t = X;
        using y_t = Y;

        using data_point = typename std::pair<x_t, y_t>;

        using convex_polygon_t = pfa::convex_polygon<polygon_t>;
        using upperbound_t = typename convex_polygon_t::upperbound_t;
        using lowerbound_t = typename convex_polygon_t::lowerbound_t;
        using boundaries_t = typename convex_polygon_t::boundaries_t;
        using point_t = typename convex_polygon_t::point_t;

        using segment_t = typename convex_polygon_t::segment_t;

        enum class approx_fun_t : uint8_t {
            Linear = 0,
            Exponential = 1,
            Quadratic = 2,
            Sqrt = 3,
            COUNT = 4
        };

    public:

        struct piecewise_linear_approximation {

            int64_t epsilon{};

            constexpr explicit piecewise_linear_approximation() : epsilon{2L} {}
            constexpr explicit piecewise_linear_approximation(const int64_t &e) : epsilon{e} {}

            boundaries_t compute_bounds(const data_point &p) const {
                assert(p.first > 0);

                auto m = -static_cast<y_t>(p.first);

                auto _uq = p.second + epsilon;
                auto _lq = p.second - epsilon;

                lowerbound_t l{typename convex_polygon_t::segment_t(m, _lq)};
                upperbound_t u{typename convex_polygon_t::segment_t(m, _uq)};
                return boundaries_t{u, l};
            }

            struct linear {

                x_t starting_position;
                T1 a;
                T2 b;
                //Segment d{};

                constexpr linear() : starting_position{0}, a{}, b{} {}
                //constexpr linear(x_t pos, const T1 &m, const T2 &q, const Segment &d_ = Segment{}) : starting_position{pos}, a(m), b{q}, d(d_) {}
                constexpr linear(x_t pos, const T1 &m, const T2 &q) : starting_position{pos}, a(m), b{q} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const double x = i - starting_position;
                    return std::ceil(a * x + b);
                }

                [[nodiscard]] constexpr static size_t lossy_size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 2; //std::log2((uint64_t) approx_fun_t::COUNT);
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 3; //std::log2((uint64_t) approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                /* copy with diagonal
                inline linear copy(x_t s1) const {
                    // f(x) = a * (x - s) + b
                    // g(x) = a' * (x - s1) + b' ^ g(x) = f(x)
                    // a' * (x - s1) + b' = a * (x - s) + b
                    // b' = a * (x - s) + b - a' * (x - s1)
                    // taking x = s1
                    // b' = a * (s1 - s) + b - a' * (s1 - s1)
                    // b' = a * (s1 - s) + b
                    // a' = a

                    auto z = static_cast<T2>(s1 - starting_position);
                    auto _a0 = d.p0().x();
                    auto _b0 = _a0 * z + d.p0().y();

                    auto _a1 = d.p1().x();
                    auto _b1 = _a1 * z + d.p1().y();

                    auto y = a * z + b;
                    return linear{s1, a, y, Segment::from_points(Point{_a0, _b0}, Point{_a1, _b1})};
                }
                */

                inline linear copy(x_t s1) const {
                    auto z = static_cast<T2>(s1 - starting_position);
                    auto y = a * z + b;

                    return linear{s1, a, static_cast<T2>(y)};
                }

                inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(std::nullopt, std::nullopt, a, b);
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Linear;
                }

                /*
                constexpr inline Segment diagonal() const {
                    return d;
                }
                */
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
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(p.second)}, Point{0.0, static_cast<T2>(p.second)});
                    //return linear{p.first, 0.0, static_cast<T2>(p.second), d};
                    return linear{p.first, static_cast<T1>(0.0), static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._s._q + l0._s._q) / 2.0;
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(b)}, Point{0.0, static_cast<T2>(b)});
                    //return linear{p.first, 0.0, static_cast<T2>(b), d};
                    return linear{p.first, static_cast<T1>(0.0), static_cast<T2>(b)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    //auto d = Segment::from_points(pl, pr);
                    //return linear{p.first, static_cast<T1>(mp_a), static_cast<T2>(mp_b), d};
                    return linear{p.first, static_cast<T1>(mp_a), static_cast<T2>(mp_b)};
                }
            }

            inline auto get_approximations(const std::vector<linear>& data, const uint32_t n) const {
                std::vector<y_t> result(n);

                for (x_t start = 0; start < data.size(); ++start) {
                    auto f = data[start];
                    auto end = (start == data.size() - 1)? n : data[start + 1].starting_position;
                    for (x_t i = f.starting_position; i < end; ++i) {
                        result[i] = f(i + 1);
                    }
                }

                return result;
            }

            template<typename It>
            inline std::vector<linear> make_approximation(It begin, It end) {

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
                        //std::cout << i << std::endl;
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

                auto m = -std::sqrt(static_cast<polygon_t>(p.first));

                auto _uq = p.second + epsilon;
                auto _lq = p.second - epsilon;

                lowerbound_t l{typename convex_polygon_t::segment_t(m, _lq)};
                upperbound_t u{typename convex_polygon_t::segment_t(m, _uq)};
                return boundaries_t{u, l};
            }

            struct sqrt {

                x_t starting_position;
                x_t d = 0;
                T1 a;
                T2 b;
                //Segment _diagonal{};

                constexpr sqrt() : starting_position{0}, a{}, b{} {}

                //constexpr sqrt(x_t pos, const T1 _a, const T2 _b, const Segment& diagonal) : starting_position{pos}, a{_a}, b{_b}, _diagonal{diagonal} {}
                constexpr sqrt(x_t pos, const T1 _a, const T2 _b) : starting_position{pos}, a{_a}, b{_b} {}
                //constexpr sqrt(x_t pos0, x_t pos1, const T1 _a, const T2 _b, const Segment &diagonal = Segment{}) : starting_position{pos0}, d{pos1}, a{_a}, b{_b}, _diagonal{diagonal} {}
                constexpr sqrt(x_t pos0, x_t pos1, const T1 _a, const T2 _b) : starting_position{pos0}, d{pos1}, a{_a}, b{_b} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const double x = i - (starting_position - d);
                    return std::round(a * std::sqrt(x) + b);
                }

                [[nodiscard]] constexpr static size_t lossy_size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 2; //std::log2((uint64_t) approx_fun_t::COUNT);
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    // we count x_t because, nonetheless we use a bv for the starting positions, we need to store the residual_ foffset for each starting_position
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 3;//std::log2((uint64_t)approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                [[nodiscard]] inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(d, std::nullopt, a, b);
                }

                inline sqrt copy(x_t s1) const {
                    auto z = s1 - starting_position;
                    //return sqrt{s1, z, a, b, _diagonal};
                    return sqrt{s1, z, a, b};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Sqrt;
                }

                /*
                constexpr inline Segment diagonal() const {
                    return _diagonal;
                }
                */
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
                    //auto _d = Segment::from_points(Point{0.0, static_cast<T2>(p.second)}, Point{0.0, static_cast<T2>(p.second)});
                    //return sqrt{p.first, 0.0, static_cast<T2>(p.second), _d};
                    return sqrt{p.first, static_cast<T1>(0.0), static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._s._q + l0._s._q) / 2.0;
                    //auto _d = Segment::from_points(Point{0.0, static_cast<T2>(b)}, Point{0.0, static_cast<T2>(b)});
                    //return sqrt{p.first, 0.0, static_cast<T2>(b), _d};
                    return sqrt{p.first, static_cast<T1>(0.0), static_cast<T2>(b)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    //auto _d = Segment::from_points(pl, pr);
                    //return sqrt{p.first, static_cast<T1>(mp_a), static_cast<T2>(mp_b), _d};
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

                typename convex_polygon_t::value_t _uq, _lq;

                _uq = std::log(static_cast<convex_polygon_t::value_t>(p.second + epsilon));
                _lq = std::log(static_cast<convex_polygon_t::value_t>(p.second - epsilon));

                lowerbound_t l{typename convex_polygon_t::segment_t(m, _lq)};
                upperbound_t u{typename convex_polygon_t::segment_t(m, _uq)};
                return {u, l};
            }

            struct exponential {

                x_t starting_position;
                T1 a;
                T2 b;
                //Segment _diagonal{};

                constexpr exponential() : starting_position{0}, a{}, b{} {}

                //constexpr exponential(x_t pos, const T1 _a, const T2 _b, const Segment &d = Segment{}) : starting_position{pos}, a{_a}, b{_b}, _diagonal{d} {}
                constexpr exponential(x_t pos, const T1 _a, const T2 _b) : starting_position{pos}, a{_a}, b{_b} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const double x = i - starting_position;
                    return round(b * std::exp(a * x));
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 3;//std::log2((uint64_t)approx_fun_t::COUNT);
                }

                [[nodiscard]] constexpr static size_t lossy_size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T2)) * 8 + 2;//std::log2((uint64_t)approx_fun_t::COUNT);
                }

                constexpr x_t get_start() const {
                    return starting_position;
                }

                [[nodiscard]] inline std::tuple<std::optional<x_t>, std::optional<T1>, T1, T2> parameters() const {
                    return std::make_tuple(std::nullopt, std::nullopt, a, b);
                }

                /* copy with diagonal
                inline exponential copy(x_t s1) const {
                    // f(x) = b * exp(a*(x - s))
                    // g(x) = b' * exp(a'*(x - s1)) and g(x) = f(x)
                    // b' * exp(a'*(x - s1)) = b * exp(a*(x - s))
                    // b' = b * exp(a*(x - s)) / exp(a'*(x - s1))
                    // taking x = s1
                    // b' = b * exp(a*(s1 - s))
                    // a' = a

                    auto z = static_cast<T2>(s1 - starting_position);
                    auto _a0 = _diagonal.p0().x();
                    auto _b0 = _diagonal.p0().y() * std::exp(_a0 * z);

                    auto _a1 = _diagonal.p1().x();
                    auto _b1 = _diagonal.p1().y() * std::exp(_a1 * z);

                    T1 a1 = a;
                    auto b1 = b * std::exp(a * z);
                    return exponential{s1, static_cast<T1>(a1), static_cast<T2>(b1), Segment::from_points(Point{_a0, _b0}, Point{_a1, _b1})};
                }
                */

                inline exponential copy(x_t s1) const {
                    auto z = static_cast<T2>(s1 - starting_position);
                    auto b1 = b * std::exp(a * z);
                    return exponential{s1, a, static_cast<T2>(b1)};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Exponential;
                }

                /*
                constexpr inline Segment diagonal() const {
                    return _diagonal;
                }
                */
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
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(p.second)}, Point{0.0, static_cast<T2>(p.second)});
                    //return exponential{p.first, 0.0, static_cast<T2>(p.second), d};
                    return exponential{p.first, static_cast<T1>(0.0), static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._s._q + l0._s._q) / 2.0;
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(b)}, Point{0.0, static_cast<T2>(b)});
                    //return exponential{p.first, 0.0, static_cast<T2>(std::exp(b)), d};
                    return exponential{p.first, static_cast<T1>(0.0), static_cast<T2>(std::exp(b))};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();


                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    //auto mp_a = pl.x();
                    //auto mp_b = pl.y();

                    //auto d = Segment::from_points(pl, pr);
                    //return exponential{p.first, static_cast<T1>(mp_a), static_cast<T2>(std::exp(mp_b)), d};
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

                auto m = -static_cast<polygon_t>(p0.first + p.first);

                auto _lq = ((p.second - epsilon) - p0.second) / static_cast<polygon_t>(p.first - p0.first);
                auto _uq = ((p.second + epsilon) - p0.second) / static_cast<polygon_t>(p.first - p0.first);

                lowerbound_t l{typename convex_polygon_t::segment_t(m, _lq)};
                upperbound_t u{typename convex_polygon_t::segment_t(m, _uq)};
                return boundaries_t{u, l};
            }

            struct quadratic {

                x_t starting_position;
                T1 a;
                T1 b;
                T2 c;
                //Segment _diagonal{};

                constexpr quadratic() : starting_position{0}, a{}, b{}, c{} {}

                //constexpr quadratic(x_t pos, const T1 _a, const T1 _b, const T2 _c, const Segment &d = Segment{}) : starting_position{pos}, a{_a}, b{_b}, c{_c}, _diagonal{d} {}
                constexpr quadratic(x_t pos, const T1 _a, const T1 _b, const T2 _c) : starting_position{pos}, a{_a}, b{_b}, c{_c} {}

                inline y_t operator()(x_t i) const {
                    assert(i >= starting_position);
                    const double x = (i - 1) - starting_position;
                    return std::ceil(a * x * x + b * x + c);
                }

                [[nodiscard]] constexpr static size_t lossy_size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + (sizeof(T1) * 2) + sizeof(T2)) * 8 + 2;//std::log2((uint64_t)approx_fun_t::COUNT);
                }

                [[nodiscard]] constexpr static size_t size_in_bits() {
                    // 3 due to the random access with the type of the function (model_types0 + model_types1 + qbv)
                    return (sizeof(x_t) + sizeof(T1) + sizeof(T1) + sizeof(T2)) * 8 + 3;//std::log2((uint64_t)approx_fun_t::COUNT);
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

                /* copy with diagonal
                inline quadratic copy(x_t s1) const {
                    // f(x) = a(x - s)^2 + b(x - s) + c
                    // g(x) = a'(x - s1)^2 + b'(x - s1) + c'
                    // g(x) = f(x) for each x >= s1 => a'(x - s1)^2 + b'(x - s1) + c' = a(x - s)^2 + b(x - s) + c
                    // Let c' = f(s1) and f(x) = g(x) for each x >= s1
                    // c' = a(s1-s)^2+b(s1-s)+ c
                    // taking x = s1 + k
                    // a'(x - s1)^2 + b'(x - s1) + c' = a(x - s)^2 + b(x - s) + c
                    // a'(k)^2 + b'(k) + c' = a(s1 + k - s)^2 + b(s1 + k - s) + c
                    // a'(k)^2 + b'(k) + a(s1-s)^2+b(s1-s)+ c = a(s1 + k - s)^2 + b(s1 + k - s) + c
                    // with k = 1 and d = s1 - s
                    // a'(1)^2 + b'(1) + a(d)^2+b(d)+ c = a(d+1)^2+b(d+1)+ c
                    // a' + b' + a(d)^2 + bd + c = a(d^2+2d+1)+ bd + b + c
                    // a' + b' + a(d)^2 = a(d)^2 + 2ad + a + b
                    // b' = 2ad + a + b - a'
                    // a' = a

                    auto z = static_cast<T2>(s1 - starting_position);
                    auto _a0 = _diagonal.p0().x();
                    auto _b0 = 2 * _a0 * z + _diagonal.p0().y();

                    auto _a1 = _diagonal.p1().x();
                    auto _b1 = 2 * _a1 * z + _diagonal.p1().y();

                    auto a1 = a;
                    auto b1 = 2 * a1 * z + b;
                    auto c1 = a1 * z * z + b * z + c;
                    return quadratic{s1, static_cast<T1>(a1), static_cast<T1>(b1), static_cast<T2>(c1), Segment::from_points(Point{_a0, _b0}, Point{_a1, _b1})};
                }*/

                inline quadratic copy(x_t s1) const {
                    auto z = static_cast<T2>(s1 - starting_position);
                    auto a1 = a;
                    auto b1 = 2 * a1 * z + b;
                    auto c1 = a1 * z * z + b * z + c;
                    return quadratic{s1, static_cast<T1>(a1), static_cast<T1>(b1), static_cast<T2>(c1)};
                }

                constexpr inline approx_fun_t type() {
                    return approx_fun_t::Quadratic;
                }

                /*
                constexpr inline Segment diagonal() const {
                    return _diagonal;
                }
                */
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
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(p.second)}, Point{0.0, static_cast<T2>(p.second)});
                    //return quadratic{p.first, 0.0, 0.0, static_cast<T2>(p.second), d};
                    return quadratic{p.first, static_cast<T1>(0.0), static_cast<T1>(0.0), static_cast<T2>(p.second)};
                } else if (g.is_init()) {
                    auto [u0, l0] = g.init.value();
                    auto b = (u0._s._q + l0._s._q) / 2.0;
                    //auto d = Segment::from_points(Point{0.0, static_cast<T2>(b)}, Point{0.0, static_cast<T2>(b)});
                    //return quadratic{p.first, 0.0, static_cast<T1>(b), static_cast<T2>(p.second), d};
                    return quadratic{p.first, static_cast<T1>(0.0), static_cast<T1>(b), static_cast<T2>(p.second)};
                } else {
                    auto pl = g.ul();
                    auto pr = g.lr();

                    auto mp_a = (pl.x() + pr.x()) / 2.0;
                    auto mp_b = (pr.y() + pl.y()) / 2.0;

                    //auto d = Segment::from_points(pl, pr);
                    //return quadratic{p.first, static_cast<T1>(mp_a), static_cast<T1>(mp_b), static_cast<T2>(p.second), d};
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
                for (x_t i = 1; i <= n; ++i) {
                    y_t last_value = *(begin + i);
                    auto dp = data_point{(i - last_starting_point.first), last_value};
                    bool intersect = add_point(g, p0, dp);
                    if (!intersect) {
                        auto f = create_fun(g, last_starting_point);
                        result.emplace_back(f);
                        g.clear();
                        last_starting_point = data_point{i, last_value};
                        p0.second = last_value;
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
        using psa_t = piecewise_sqrt_approximation;
        using pea_t = piecewise_exponential_approximation;
        using pqa_t = piecewise_quadratic_approximation;

        using pna_t = std::variant<pla_t, pea_t, pqa_t, psa_t>;
        using vec_pna_t = typename std::vector<pna_t>;
        using pna_fun_t = std::variant<typename pla_t::fun_t, typename pea_t::fun_t, typename pqa_t::fun_t, typename psa_t::fun_t>;

        static pna_t make_model(approx_fun_t mt, int64_t epsilon) {
            switch (mt) {
                case approx_fun_t::Linear:
                    return pla_t{epsilon};
                case approx_fun_t::Quadratic:
                    return pqa_t{epsilon};
                case approx_fun_t::Sqrt:
                    return psa_t{epsilon};
                case approx_fun_t::Exponential:
                    return pea_t{epsilon};
                default:
                    throw std::runtime_error("Not implemented");
            }
        }

        struct piecewise_non_linear_approximation {

            static inline auto make_fun(approx_fun_t mt, x_t start_pos, std::optional<x_t> d, std::optional<T1> t0, T1 t1, T2 t2) {
                switch (mt) {
                    case approx_fun_t::Linear:
                        assert(!d.has_value());
                        return pna_fun_t{typename pla_t::fun_t{start_pos, t1, t2}};
                    case approx_fun_t::Quadratic:
                        assert(!d.has_value());
                        assert(t0.has_value());
                        return pna_fun_t{typename pqa_t::fun_t{start_pos, t0.value(), t1, t2}};
                    case approx_fun_t::Sqrt:
                        assert(d.has_value());
                        return pna_fun_t{typename psa_t::fun_t{start_pos, d.value(), t1, t2}};
                    case approx_fun_t::Exponential:
                        assert(!d.has_value());
                        return pna_fun_t{typename pea_t::fun_t{start_pos, t1, t2}};
                    default:
                        throw std::runtime_error("Not implemented");
                }
            }
        };

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

            using pfa_tuple_t = std::tuple<Pfa...>;

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


    };
}