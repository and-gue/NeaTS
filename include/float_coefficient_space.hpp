#pragma once

#include <tuple>
#include <numeric>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <optional>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/numeric/interval.hpp>
#include <variant>
#include <algorithm>

namespace fa::fcs {

    template<class... Ts>
    struct match : Ts ... {
        using Ts::operator()...;
    };

    template<class... Ts>
    match(Ts...) -> match<Ts...>;

    template<typename T>
    struct convex_polygon {
        using point_t = ::boost::geometry::model::d2::point_xy<T>;
        using value_t = T;

        //inline static std::vector<point_t> out_intersections{};

        struct segment_t {
            T _m;
            T _q;

            T _x0;
            T _x1;

            std::optional<std::pair<T, T>> y_range = std::nullopt; // this segment is vertical or a single point

            constexpr segment_t() = default;

            constexpr segment_t(const T m, const T q,
                                const T x0 = std::numeric_limits<T>::lowest(),
                                const T x1 = std::numeric_limits<T>::max()) : _m{m}, _q{q}, _x0{x0}, _x1{x1} {
                //assert(segment_t::eq(x0, x1, 25) || x0 < x1);
            }

            constexpr static inline segment_t vertical(const T x, const T y0, const T y1) {
                segment_t s;
                s._m = {};
                s._q = {};
                s._x0 = x;
                s._x1 = x;
                s.y_range = std::make_pair(y0, y1);
                return s;
            }

            /*
            static inline bool eq(T x, T y, uint32_t ulp = 1) {
                static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
                using I = std::conditional_t<sizeof(T) <= 4, int32_t, int64_t>;
                I xi = *(I *) &x;
                if (xi < 0) xi = std::numeric_limits<I>::max() - xi;

                I yi = *(I *) &y;
                if (yi < 0) yi = std::numeric_limits<I>::max() - yi;

                return std::make_unsigned_t<I>(std::abs(xi - yi)) <= ulp;
            }
            */

            /*
            static inline bool eq(T x, T x_) {
                static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
                T precision;
                if constexpr (std::same_as<T, double>) precision = 1e-13;
                else {
                    static_assert(std::same_as<T, long double>, "T must be a floating point type");
                    precision = 1e-16;
                }
                if (x == 0.0 || x_ == 0.0) return fabs(x - x_) < precision;
                return (x == x_) || (fabs(x - x_) < precision * fabs(x));
            }
            */

            static inline bool eq(T x, T x_, T max_abs_diff = 1e-9, T max_rel_diff = 1e-13) {
                if constexpr (sizeof(T) > 8) {
                    max_abs_diff = 1e-19;
                    max_rel_diff = 1e-19;
                }
                // Check if the numbers are really close -- needed
                // when comparing numbers near zero.
                auto diff = std::abs(x - x_);
                if (diff <= max_abs_diff) return true;

                //x = std::abs(x);
                //x_ = std::abs(x_);
                auto largest = (x > x_) ? std::abs(x) : std::abs(x_);

                if (diff <= largest * max_rel_diff) return true;
                return false;
            }

            static inline segment_t from_points(const point_t &p0, const point_t &p1) {

                /*
                if (eq(p0.x(), p1.x())) {
                    return segment_t{0, std::, p0.x(), p1.x()};
                    //return segment_t::vertical(p0.x(), std::min(p0.y(), p1.y()), std::max(p0.y(), p1.y()));
                }
                */

                if (p1.x() == p0.x()) {
                    //assert(p0.y() == p1.y());
                    return segment_t{0, p0.y(), p0.x(), p1.x()};
                }

                auto m = (p1.y() - p0.y()) / (p1.x() - p0.x());
                auto q = p0.y() - m * p0.x();

                return segment_t{m, q, p0.x(), p1.x()};
            }

            inline point_t p0() const {
                if (y_range.has_value()) return point_t{_x0, y_range.value().first};
                else return point_t{_x0, _m * _x0 + _q};
            }

            inline point_t p1() const {
                if (y_range.has_value()) return point_t{_x1, y_range.value().second};
                else return point_t{_x1, _m * _x1 + _q};
            }

            inline bool intersects(const segment_t &s) const {

                auto y0 = s._m * _x0 + s._q;
                auto y1 = s._m * _x1 + s._q;

                auto p0y = _m * _x0 + _q;
                auto p1y = _m * _x1 + _q;

                //return y0 >= p0y && y1 <= p1y;
                return (eq(y0, p0y) || y0 > p0y) && (eq(y1, p1y) || y1 < p1y);

                //auto s0 = ::boost::geometry::model::segment<point_t>{point_t{_x0, y0}, point_t{_x1, y1}};
                //auto s1 = ::boost::geometry::model::segment<point_t>{point_t{_x0, p0y}, point_t{_x1, p1y}};
                //return ::boost::geometry::intersects(s0, s1);

            }

            /*
            static inline bool less(const segment_t&s0, const segment_t&s1) const {
                // s0 is data[i] and s1 is seg._s
                //assert(s0._x0 >= s1._x0 and s0._x1 <= s1._x1);

                auto x0 = s0._x0;
                auto x1 = s0._x1;

                auto s0_left = s0._m * x0 + s0._q;
                auto s0_right = s0._m * x1 + s0._q;

                auto s1_left = s1._m * x0 + s1._q;
                auto s1_right = s1._m * _x1 + s1._q;

                //return s0_left < s1_left && s0_right < s1_right;
                return (!eq(s0_left, s1_left) && s0_left < s1_left) && (!eq(s0_right, s1_right) && s0_right < s1_right);
            }
            */

            bool operator<(const segment_t &s) const {
                auto sy0 = s._m * _x0 + s._q;
                auto sy1 = s._m * _x1 + s._q;

                auto py0 = _m * _x0 + _q;
                auto py1 = _m * _x1 + _q;

                //return sy0 > py0 && sy1 > py1;
                return (sy0 > py0) & !eq(sy0, py0) && (sy1 > py1) & !eq(sy1, py1);
            }

            template<bool Upper>
            inline segment_t intersection(const segment_t &s) const {
                T x, y;

                if (segment_t::eq(_m, s._m)) {
                    assert(segment_t::eq(_q, s._q));
                    return segment_t{0, p0().y(), p0().x(), p1().x()};
                } else {
                    x = (_q - s._q) / (s._m - _m);
                    y = _m * x + _q;
                }

                if constexpr (Upper) {
                    if (x == p0().x())
                        return segment_t{0, y, x, p0().x()};
                    auto m = (y - p0().y()) / (x - p0().x());
                    auto q = y - m * x;
                    return segment_t{m, q, p0().x(), x};
                } else {
                    if (x == p1().x())
                        return segment_t{0, y, x, p1().x()};
                    auto m = (y - p1().y()) / (x - p1().x());
                    auto q = y - m * x;
                    return segment_t{m, q, x, p1().x()};
                }
            }
        };

        std::vector<segment_t> upper{};
        uint32_t up_start{0};
        std::vector<segment_t> lower{};
        uint32_t lo_start{0};

    public:

        template<bool upper = true>
        struct halfplane {
            segment_t _s;

            constexpr halfplane() = default;

            constexpr halfplane(const segment_t &s) : _s{s} {}

            [[nodiscard]] constexpr bool is_upper() const {
                return upper;
            }
        };

        using upperbound_t = halfplane<true>;
        using lowerbound_t = halfplane<false>;
        using boundaries_t = std::pair<upperbound_t, lowerbound_t>;

        std::optional<boundaries_t> init = std::nullopt;

        constexpr convex_polygon() {
            upper.reserve(1 << 18);
            lower.reserve(1 << 18);
            assert(empty());
        }

        /*
        inline size_t search_upper_intersection(auto *data, size_t len, const auto &seg) const {
            auto i = 0;
            while (i < (len - 1)) {
                if (data[i] < seg._s) ++i;
                else {
                    //assert(data[i].intersects(seg._s));
                    return i;
                };
            }
            return len - 1;
        }

        inline size_t search_lower_intersection(auto *data, size_t len, const auto &seg) const {
            auto i = len - 1;
            while (i > 0) {
                if (data[i] < seg._s) --i;
                else {
                    //assert(data[i].intersects(seg._s));
                    return i;
                };
            }
            return 0;
        }
        */

        /*
        template<typename It, class S>
        It branchless_search(It first, It last, const S& x) const {
            int len = std::distance(first, last);
            It base = first;
            while (len > 1) {
                int half = len / 2;
                len -= half;
                __builtin_prefetch(&base[len / 2 - 1]);
                __builtin_prefetch(&base[half + len / 2 - 1]);
                base += (base[half - 1] < x) * half;
            }
            return base + (*base < x);
        }
        */

        template<typename It>
        inline size_t search_intersection(It first, It last, const auto &seg) const {
            auto it = std::lower_bound(first, last, seg._s);
            auto r = std::distance(first, it);
            //assert(it->intersects(seg._s));
            return r == std::distance(first, last) ? r - 1 : r;
        }

        template<bool Upper>
        inline auto cut(const halfplane<Upper> &u) const {

            auto index_seg_u = search_intersection(upper.begin() + up_start, upper.end(), u);
            auto index_seg_l = search_intersection(lower.rbegin(), lower.rend() - lo_start, u);
            index_seg_l = (lower.size() - 1 - lo_start) - index_seg_l;

            auto new_seg_u = upper[index_seg_u + up_start].template intersection<Upper>(u._s);
            auto new_seg_l = lower[index_seg_l + lo_start].template intersection<Upper>(u._s);

            return std::make_tuple(index_seg_u, index_seg_l, new_seg_u, new_seg_l);
        }

        inline std::variant<std::tuple<size_t, size_t, segment_t, segment_t>, bool> cut_with_upper_bound(
                const upperbound_t &u) {

            auto pr = lower[lo_start].p1();
            auto pl = upper[up_start].p0();

            auto narrow_u = u._s;
            narrow_u._x0 = pl.x();
            narrow_u._x1 = pr.x();

            auto ul = narrow_u.p0().y();
            auto ur = narrow_u.p1().y();

            if (ur >= pr.y()) {
                return true;
            } else if (ul >= pl.y() && ur < pr.y()) {
                auto t = cut(std::decay_t<decltype(u)>{narrow_u});

                auto new_seg_u = std::get<2>(t);
                auto new_seg_l = std::get<3>(t);

                //assert(new_seg_u.p0().x() <= new_seg_l.p0().x());
                //assert(new_seg_u.p1().x() <= new_seg_l.p1().x());

                //assert(new_seg_u.p0().y() >= new_seg_l.p0().y());
                //assert(new_seg_u.p1().y() >= new_seg_l.p1().y());

                return t;
            } else {
                return false;
            }
        }

        inline std::variant<std::tuple<size_t, size_t, segment_t, segment_t>, bool> cut_with_lower_bound(
                const lowerbound_t &l) {

            auto pl = upper[up_start].p0();
            auto pr = lower[lo_start].p1();

            auto narrow_l = l._s;
            narrow_l._x0 = pl.x();
            narrow_l._x1 = pr.x();

            auto ll = narrow_l.p0().y();
            auto lr = narrow_l.p1().y();

            if (ll <= pl.y()) {
                return true;
            } else if (ll > pl.y() && lr <= pr.y()) {
                auto t = cut(std::decay_t<decltype(l)>{narrow_l});

                auto new_seg_u = std::get<2>(t);
                auto new_seg_l = std::get<3>(t);

                //assert(new_seg_u.p0().x() <= new_seg_l.p0().x());
                //assert(new_seg_u.p1().x() <= new_seg_l.p1().x());

                //assert(new_seg_u.p0().y() >= new_seg_l.p0().y());
                //assert(new_seg_u.p1().y() >= new_seg_l.p1().y());

                return t;
            } else {
                //assert(l.right().second.y() > d.second.y());
                return false;
            }
        }

        inline bool update(const upperbound_t &u, const lowerbound_t &l) {
            //assert(u1.sign() == 0 && l1.sign() == 1);
            if (empty()) {
                init = boundaries_t{u, l};
                return true;
            } else if (is_init()) {

                auto [s_u0, s_l0] = init.value();

                auto u0 = (s_u0._s.template intersection<false>(l._s)).template intersection<true>(u._s);
                auto l1 = (s_l0._s.template intersection<false>(l._s)).template intersection<true>(u._s);

                auto u1 = segment_t::from_points(u0.p1(), l1.p1());
                //assert(su2.p0() == su1.p1() && su2.p1() == sl1.p1());
                auto l0 = segment_t::from_points(u0.p0(), l1.p0());
                //assert(sl2.p0() == su1.p0() && sl2.p1() == sl1.p0());

                upper.push_back(u0);
                upper.push_back(u1);

                lower.push_back(l1);
                lower.push_back(l0);

                init = std::nullopt;
                return true;
            } else {
                auto vl = cut_with_lower_bound(l); // if t then create pl -- lower[lo_start] from points

                bool res_l;
                std::visit(match{
                        [&](std::tuple<size_t, size_t, segment_t, segment_t> a1) {
                            const auto &[i1, j1, su1, sl1] = a1;
                            up_start += i1;
                            upper[up_start] = su1;

                            const auto new_lower_edge = segment_t::from_points(su1.p0(), sl1.p0());
                            lower.resize(j1 + lo_start);
                            //if (!segment_t::eq(sl1._x0, sl1._x1, 5))
                            lower.push_back(sl1);
                            lower.push_back(new_lower_edge);
                            res_l = true;
                        },
                        [&res_l](bool a1) { res_l = a1; },
                }, vl);

                if (res_l) {
                    auto vu = cut_with_upper_bound(u);
                    bool res_u;

                    std::visit(match{
                            [&](std::tuple<size_t, size_t, segment_t, segment_t> a2) {
                                const auto &[i2, j2, su2, sl2] = a2;
                                upper.resize(i2 + up_start);
                                //if (!segment_t::eq(su2._x0, su2._x1, 5))
                                upper.push_back(su2);
                                const auto new_upper_edge = segment_t::from_points(su2.p1(), sl2.p1());
                                upper.push_back(new_upper_edge);

                                lo_start += j2;
                                lower[lo_start] = sl2;
                                res_u = true;
                            },
                            [&res_u](bool a2) { res_u = a2; }
                    }, vu);

                    return res_u;
                }
                return false;
            }
        }

        [[nodiscard]] constexpr bool empty() const {
            return upper.empty() && lower.empty() && !is_init();
        }

        inline void clear() {
            upper.clear();
            lower.clear();
            lo_start = 0;
            up_start = 0;
            //edges.clear();
            //out_intersections.clear();
            init = std::nullopt;
        }

        inline point_t ul() const {
            return upper[up_start].p0();
        }

        inline point_t lr() const {
            return lower[lo_start].p1();
        }

        [[nodiscard]] inline bool is_init() const {
            return init.has_value();
        }

    };
}