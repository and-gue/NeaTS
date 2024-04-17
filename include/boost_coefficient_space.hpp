#pragma once

#include <deque>
#include <tuple>
#include <numeric>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <optional>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <variant>

namespace fa::boost::fcs {

    template<typename T>
    class interval {
        static_assert(std::is_floating_point_v<T>, "T must be floating point");
        using I = int64_t;
        using U = std::make_unsigned<I>;

        T _x0;
        T _x1;

        static inline T succ(T x) {
            return std::nextafter(x, std::numeric_limits<T>::max());
        }

        static inline T pred(T x) {
            return std::nextafter(x, std::numeric_limits<T>::lowest());
        }

    public:

        using value_t = T;
        static constexpr value_t zero = value_t{0.0};

        static auto ulp_diff(T x, T y) {
            static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
            I xi = *(I *) &x;
            if (xi < 0) xi = std::numeric_limits<I>::max() - xi;

            I yi = *(I *) &y;
            if (yi < 0) yi = std::numeric_limits<I>::max() - yi;

            return static_cast<std::make_unsigned_t<I>>(std::abs(xi - yi));
        }

        static auto ulp_sum(T x, auto y) {
            static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
            static_assert(std::is_integral<decltype(y)>::value, "y must be an integral type");
            static_assert(std::is_unsigned<decltype(y)>::value, "y must be an unsigned type");
            I xi = *(I *) &x;
            if (xi < 0) xi = std::numeric_limits<I>::max() - xi;

            I sum = xi - y;
            return *(T *) &(sum);
        }

        constexpr static interval whole() {
            return interval<T>{
                    succ(std::numeric_limits<T>::lowest()),
                    prev(std::numeric_limits<T>::max())
            };
        }

        constexpr explicit interval(T x) : _x0{pred(x)}, _x1{succ(x)} {}

        constexpr explicit interval() : _x0{value_t{+0.0}}, _x1{value_t{-0.0}} {}

        // true if the intervals have the same bits representation
        inline bool operator==(const interval &d) const {
            return (_x0 == d._x0) && (_x1 == d._x1);
        }

        inline bool operator<(const interval &d) const {
            return _x1 < d._x0;
        }

        inline bool operator>(const interval &d) const {
            return d < (*this);
        }

        inline bool operator<=(const interval &d) const {
            return *(this) < d || *(this) == d;
        }

        inline bool operator>=(const interval &d) const {
            return d <= (*this);
        }

        inline bool operator!=(const interval &d) const {
            return !(*(this) == d);
        }

        interval operator+(const interval &d) const {
            const T x0 = pred(_x0 + d._x0);
            const T x1 = succ(_x1 + d._x1);
            return interval{x0, x1};
        }

        interval operator-() const {
            auto x0 = -_x0;
            auto x1 = -_x1;
            if (x0 > x1) return interval{x1, x0};
            else return interval{x0, x1};
        }

        interval operator-(const interval &d) const {
            const T x0 = pred(_x0 - d._x1);
            const T x1 = succ(_x1 - d._x0);
            return interval{x0, x1};
        }

        interval operator*(const interval &d) const {
            const T &xl = _x0;
            const T &xu = _x1;
            const T &yl = d._x0;
            const T &yu = d._x1;

            if (is_neg(xl))
                if (is_pos(xu))
                    if (is_neg(yl))
                        if (is_pos(yu)) // M * M
                            return interval{std::min(pred(xl * yu), pred(xu * yl)),
                                            std::max(succ(xl * yl), succ(xu * yu))};
                        else                    // M * N
                            return interval{pred(xu * yl), succ(xl * yl)};
                    else if (is_pos(yu)) // M * P
                        return interval{pred(xl * yu), succ(xu * yu)};
                    else                    // M * Z
                        return interval{};
                else if (is_neg(yl))
                    if (is_pos(yu)) // N * M
                        return interval{pred(xl * yu), succ(xl * yl)};
                    else                    // N * N
                        return interval{pred(xu * yu), succ(xl * yl)};
                else if (is_pos(yu)) // N * P
                    return interval{pred(xl * yu), succ(xu * yl)};
                else                    // N * Z
                    return interval{};
            else if (is_pos(xu))
                if (is_neg(yl))
                    if (is_pos(yu)) // P * M
                        return interval{pred(xu * yl), succ(xu * yu)};
                    else // P * N
                        return interval{pred(xl * yu), succ(xu * yu)};
                else if (is_pos(yu)) // P * P
                    return interval{pred(xl * yl), succ(xu * yu)};
                else // P * Z
                    return interval{};
            else                        // Z * ?
                return interval{};
        }

        /*
        static inline doubleb div_zero(const doubleb& x) {
            if (is_zero(x._x0) && is_zero(x._x1)) return x;
            else return doubleb{succ(std::numeric_limits<T>::lowest()), pred(std::numeric_limits<T>::max())};
        }


        static inline interval div_neg(const interval &x, const T &y) {
            T xl = x._x0;
            T xu = x._x1;
            if (is_zero(xl) && is_zero(xu)) return x;
            else if (is_neg(xu)) return interval{pred(xu / y), succ(std::numeric_limits<T>::lowest())};
            else if (is_neg(xl))
                return whole();
            else return interval{succ(std::numeric_limits<T>::lowest()), succ(xl, y)};
        }

        static inline interval div_pos(const interval &x, const T &y) {
            T xl = x._x0;
            T xu = x._x1;
            if (is_zero(xl) && is_zero(xu)) return x;
            else if (is_neg(xu)) return interval{succ(std::numeric_limits<T>::lowest()), succ(xu / y),};
            else if (is_neg(xl))
                return whole();
            else return interval{prev(xl, y), pred(std::numeric_limits<T>::max())};
        }
        */

        static inline interval div_nonzero(const interval &x, const interval &y) {
            T xl = x._x0;
            T xu = x._x1;
            T yl = y._x0;
            T yu = y._x1;

            if (is_neg(xu))
                if (is_neg(yu))
                    return interval{pred(xu / yl), succ(xl / yu)};
                else
                    return interval{pred(xl / yl), succ(xu / yu)};
            else if (is_neg(xl))
                if (is_neg(yu))
                    return interval{pred(xu / yu), succ(xl / yu)};
                else
                    return interval{pred(xl / yl), succ(xu / yl)};
            else if (is_neg(yu))
                return interval{pred(xu / yu), succ(xl / yl)};
            else
                return interval{pred(xl / yu), succ(xu / yl)};
        }

        interval operator/(const interval &d) const {
            if (zero_in(d)) {
                /*
                if (!is_zero(d._x0) && is_zero(d._x1)) {
                    return div_neg(*this, d._x0);
                }
                if (is_zero(d._x0) && !is_zero(d._x1)) {
                    return div_pos(*this, d._x1);
                }
                */
                throw std::runtime_error("Division by zero");
            } else {
                return div_nonzero(*(this), d);
            }
        }


        static inline interval log(const interval &d) {
            assert(d.min() > zero);
            auto x0 = pred(std::log(d._x0));
            auto x1 = succ(std::log(d._x1));
            return interval{x0, x1};
        }

        static inline interval sqrt(const interval &d) {
            assert(d.min() > zero);
            auto x0 = pred(std::sqrt(d._x0));
            auto x1 = succ(std::sqrt(d._x1));
            return interval{x0, x1};
        }

        [[nodiscard]] inline value_t min() const {
            assert(_x0 <= _x1);
            return _x0;
        }

        [[nodiscard]] inline value_t max() const {
            assert(_x0 <= _x1);
            return _x1;
        }

        /*
        static constexpr bool ulp_eq(value_t x, value_t y, uint32_t ulp = 1) {
            assert(ulp > 0 && ulp < 4 * 1024 * 1024);
            return ulp_diff(x, y) <= ulp;
        }
        */

        constexpr interval(T x0, T x1) : _x0{x0}, _x1{x1} {}

        static inline bool is_neg(T x) {
            return x < zero;
        }

        static inline bool is_pos(T x) {
            return x > zero;
        }

        static inline bool is_zero(T x) {
            return x == zero;
        }

        static inline bool zero_in(interval v) {
            return !is_pos(v._x0) && !is_neg(v._x1);
        }

    };

    template<typename T>
    struct convex_polygon {

        using value_t = T;
        using point_t = ::boost::geometry::model::d2::point_xy<T>;
        using segment_t = ::boost::geometry::model::segment<point_t>;

        using interval_t = interval<value_t>;

        std::vector<point_t> upper{};
        uint32_t up_start{0};
        std::vector<point_t> lower{};
        uint32_t lo_start{0};

        inline static std::vector<point_t> out_intersection{};

    public:

        template<bool upper = true>
        struct halfplane {
            //static inline constexpr value_t x0 = -(1L << 33); //-(1L << 33);
            //static inline constexpr value_t x1 = (1L << 33); //(1L << 33);

            segment_t l;
            segment_t r;
            std::optional<segment_t> p = std::nullopt;

            [[nodiscard]] constexpr bool is_upper() const {
                return upper;
            }

            inline segment_t left() const {
                //assert(::boost::geometry::intersects(l, r));
                if (p.has_value()) return p.value();
                else return l;
            }

            inline segment_t right() const {
                //assert(::boost::geometry::intersects(l, r));
                if (p.has_value()) return p.value();
                else return r;
            }

            inline value_t intercept() const {
                assert(!p.has_value());
                return l.second.y();
            }

            inline point_t intersection_point() const {
                //assert(r.first.y() == l.first.y());
                assert(!p.has_value());
                return l.second;
            }

            constexpr halfplane() = default;

            /*
            constexpr halfplane(const interval_t &m, const interval_t &q) {
                if constexpr (upper) {
                    auto _ml = m.max();
                    auto _mr = m.min();
                    auto _q = q.min();

                    l = segment_t{point_t{x0, _ml * x0 + _q}, point_t{interval_t::zero, _q}};
                    r = segment_t{point_t{interval_t::zero, _q}, point_t{x1, _mr * x1 + _q}};
                    assert(::boost::geometry::intersects(l, r));
                } else {
                    auto _ml = m.min();
                    auto _mr = m.max();
                    auto _q = q.max();

                    l = segment_t{point_t{x0, _ml * x0 + _q}, point_t{interval_t::zero, _q}};
                    r = segment_t{point_t{interval_t::zero, _q}, point_t{x1, _mr * x1 + _q}};
                    assert(::boost::geometry::intersects(l, r));
                }
            }
            */

            constexpr explicit halfplane(const segment_t &s) {
                //assert(s.first.x() <= s.second.x());
                p = s;
            }

            constexpr halfplane(const segment_t &left, const segment_t &right) {
                assert(left.first.x() < interval_t::zero && right.second.x() > interval_t::zero);
                l = left;
                r = right;
                //assert(::boost::geometry::intersects(l, r));
            }

            inline bool intersects(const segment_t &segment) const {
                if (p.has_value()) {
                    return ::boost::geometry::intersects(p.value(), segment);
                } else {
                    return ::boost::geometry::intersects(l, segment) || ::boost::geometry::intersects(r, segment);
                }
            }

            inline point_t up_intersection(const segment_t &segment) const {
                assert(p.has_value());
                auto index_last_intersection = out_intersection.size() - 1;

                if (!::boost::geometry::intersects(p.value(), segment)) {
                    return segment.second;
                }

                ::boost::geometry::intersection(p.value(), segment, out_intersection);
                point_t res;
                res = out_intersection[index_last_intersection + 1];
                if (res.x() < segment.first.x()) {
                    return segment.first;
                }

                if (res.x() > segment.second.x()) {
                    return segment.second;
                }

                if (res.y() > segment.first.y()) {
                    ::boost::geometry::set<1>(res, segment.first.y());
                }

                if (res.y() < segment.second.y()) {
                    ::boost::geometry::set<1>(res, segment.second.y());
                }

                return res;
            }


            inline point_t lo_intersection(const segment_t &segment) const {
                assert(p.has_value());
                auto index_last_intersection = out_intersection.size() - 1;

                if (!::boost::geometry::intersects(p.value(), segment)) {
                    return segment.second;
                }

                ::boost::geometry::intersection(p.value(), segment, out_intersection);
                point_t res;
                res = out_intersection.back();

                if (res.x() < segment.first.x()) {
                    return segment.first;
                }

                if (res.x() > segment.second.x()) {
                    return segment.second;
                }

                if (res.y() > segment.first.y()) {
                    ::boost::geometry::set<1>(res, segment.first.y());
                }

                if (res.y() < segment.second.y()) {
                    ::boost::geometry::set<1>(res, segment.second.y());
                }

                return res;
            }

            inline halfplane<upper> split(const point_t &pi, const point_t &pj) const {
                assert(pi.x() <= pj.x() && pi.x() >= left().first.x() && pj.x() <= right().second.x());
                assert(left().first.y() >= right().second.y());
                assert(pi.y() <= left().first.y() && pj.y() >= right().second.y());
                assert(pi.y() >= pj.y());

                auto index_last_intersection = out_intersection.size() - 1;
                auto ymax = left().first.y();
                auto ymin = right().second.y();

                value_t p0y, p1y;

                auto l0 = segment_t{point_t{pi.x(), ymax}, point_t{pi.x(), ymin}};
                auto l1 = segment_t{point_t{pj.x(), ymax}, point_t{pj.x(), ymin}};

                auto fix = [](auto py, auto _ymax, auto _ymin) {
                    if (py > _ymax)
                        return _ymax;
                    else if (py < _ymin)
                        return _ymin;
                    else return py;
                };

                assert(this->intersects(l0) && this->intersects(l1));
                if (::boost::geometry::intersects(left(), l0)) {
                    ::boost::geometry::intersection(left(), l0, out_intersection);
                    p0y = fix(out_intersection[index_last_intersection + 1].y(), left().first.y(), left().second.y());
                    assert(p0y >= ymin && p0y <= ymax);
                } else {
                    assert(::boost::geometry::intersects(right(), l0));
                    ::boost::geometry::intersection(right(), l0, out_intersection);
                    p0y = fix(out_intersection.back().y(), right().first.y(), right().second.y());
                    assert(p0y >= ymin && p0y <= ymax);
                }

                index_last_intersection = out_intersection.size() - 1;
                if (::boost::geometry::intersects(right(), l1)) {
                    ::boost::geometry::intersection(right(), l1, out_intersection);
                    p1y = fix(out_intersection.back().y(), right().first.y(), right().second.y());
                    assert(p1y >= ymin && p1y <= ymax);
                } else {
                    assert(::boost::geometry::intersects(left(), l1));
                    ::boost::geometry::intersection(left(), l1, out_intersection);
                    p1y = fix(out_intersection[index_last_intersection + 1].y(), left().first.y(), left().second.y());
                    assert(p1y >= ymin && p1y <= ymax);
                }

                assert(p0y >= p1y);
                if (pi.x() <= interval_t::zero && pj.x() <= interval_t::zero) {
                    return halfplane<upper>{segment_t{point_t{pi.x(), p0y}, point_t{pj.x(), p1y}}};
                } else {
                    if (pi.x() >= interval_t::zero && pj.x() >= interval_t::zero)
                        return halfplane<upper>{segment_t{point_t{pi.x(), p0y}, point_t{pj.x(), p1y}}};
                    else
                        return halfplane<upper>{segment_t{point_t{pi.x(), p0y}, intersection_point()},
                                                segment_t{intersection_point(), point_t{pj.x(), p1y}}};
                }
            }
        };

        template<bool upper = true>
        struct halfplane_equation {

            value_t _ml;
            value_t _mr;
            value_t _q;


            constexpr halfplane_equation() = default;

            constexpr halfplane_equation(const interval_t &m, const interval_t &q) {
                if constexpr (upper) {
                    _ml = m.max();
                    _mr = m.min();
                    _q = q.min();
                } else {
                    _ml = m.min();
                    _mr = m.max();
                    _q = q.max();
                }
            }


            [[nodiscard]] constexpr bool is_upper() const {
                return upper;
            }

            inline auto intersection(const auto &h) const {

                auto px = (h._q - _q) / (_ml - h._ml);
                auto py = _ml * px + _q;
                if (px > interval_t::zero) {
                    px = (h._q - _q) / (_mr - h._mr);
                    py = _mr * px + _q;
                }
                return point_t{px, py};

            }

            inline halfplane<upper> split(const auto px0, const auto px1) {
                //assert(px0 <= px1);
                if (px1 <= interval_t::zero) {
                    return halfplane<upper>{segment_t{point_t{px0, _ml * px0 + _q}, point_t{px1, _ml * px1 + _q}}};
                } else if (px0 >= interval_t::zero) {
                    return halfplane<upper>{segment_t{point_t{px0, _mr * px0 + _q}, point_t{px1, _mr * px1 + _q}}};
                } else {
                    return halfplane<upper>{segment_t{point_t{px0, _ml * px0 + _q}, point_t{interval_t::zero, _q}},
                                            segment_t{point_t{interval_t::zero, _q}, point_t{px1, _mr * px1 + _q}}};
                    //return segment_t{point_t{px0, _ml * px0 + _q}, point_t{px1, _mr * px1 + _q}};
                }
            }
        };

        using upperbound_t = halfplane<true>;
        using lowerbound_t = halfplane<false>;
        using boundaries_t = std::pair<upperbound_t, lowerbound_t>;

        using upperbound_eq_t = halfplane_equation<true>;
        using lowerbound_eq_t = halfplane_equation<false>;
        using boundaries_eq_t = std::pair<upperbound_eq_t, lowerbound_eq_t>;

        std::optional<boundaries_eq_t> init = std::nullopt;

        constexpr convex_polygon() {
            upper.reserve(1 << 16);
            lower.reserve(1 << 16);

            out_intersection.reserve(1 << 16);
            assert(empty());
        }

        inline auto search_upper_intersection(const auto &line) {
            using halfplane_t = std::decay<decltype(line)>::type;
            point_t *data, *end, *p;
            uint32_t len;

            data = upper.data() + up_start;
            len = upper.size() - up_start;
            end = upper.data() + (upper.size() - 1);
            p = lower.data() + lo_start;

            point_t *pi, *pj;
            halfplane_t b;
            while (len > 1) {
                auto half = (len / 2);
                pi = data + (half - 1);
                pj = data + half;

                b = line.split(*pi, *pj);
                //auto intersects = pi->y() < b.left().first.y() && pj->y() >= b.right().second.y();
                auto intersects = b.intersects(segment_t{*pi, *pj});

                len -= half;
                len -= intersects * len;

                __builtin_prefetch(&data[half - 1]);
                __builtin_prefetch(&data[half]);

                __builtin_prefetch(&data[(half + 1) + (len / 2 - 1)]);
                __builtin_prefetch(&data[(half + 1) + (len / 2)]);

                data += !intersects * (pi->y() < b.left().first.y() && pj->y() < b.right().second.y()) * half;
            }

            if (data == end) {
                b = line.split(*data, *p);
                //assert(b.intersects(segment_t{*data, *p}));
                return std::make_tuple(b, data, p);
            } else {
                return std::make_tuple(b, pi, pj);
            }
        }

        inline auto search_lower_intersection(const auto &line) {
            using halfplane_t = std::decay<decltype(line)>::type;
            point_t *data, *end, *p;
            uint32_t len;

            data = lower.data() + lo_start;
            len = lower.size() - lo_start;
            end = lower.data() + (lower.size() - 1);
            p = upper.data() + up_start;

            point_t *pi, *pj;
            halfplane_t b;
            while (len > 1) {
                auto half = (len / 2);
                pi = data + half;
                pj = data + (half - 1);

                b = line.split(*pi, *pj);
                //auto intersects = pi->y() < b.left().first.y() && pj->y() >= b.right().second.y();
                auto intersects = b.intersects(segment_t{*pi, *pj});

                len -= half;
                len -= intersects * len;

                __builtin_prefetch(&data[half - 1]);
                __builtin_prefetch(&data[half]);

                __builtin_prefetch(&data[(half + 1) + (len / 2 - 1)]);
                __builtin_prefetch(&data[(half + 1) + (len / 2)]);

                data += !intersects * (pi->y() >= b.left().first.y() && pj->y() >= b.right().second.y()) * half;
            }

            if (data == end) {
                b = line.split(*p, *data);
                //assert(b.intersects(segment_t{*p, *data}));
                return std::make_tuple(b, p, data);
            } else {
                return std::make_tuple(b, pi, pj);
            }
        }


        inline void \(const upperbound_t &u) {
            auto su = search_upper_intersection(u);
            auto sl = search_lower_intersection(u);

            auto new_p0 = std::get<0>(su).up_intersection(segment_t{*std::get<1>(su), *std::get<2>(su)});
            auto new_pr = std::get<0>(sl).lo_intersection(segment_t{*std::get<1>(sl), *std::get<2>(sl)});

            if (new_pr.x() < new_p0.x()) {
                //new_pr = new_p0;
                ::boost::geometry::set<0>(new_pr, new_p0.x());
            }

            if (new_pr.y() > new_p0.y()) {
                //new_p0 = new_pr;
                ::boost::geometry::set<1>(new_pr, new_p0.y());
            }

            upper.resize(upper.size() - (&upper.back() - std::get<1>(su)));
            upper.emplace_back(new_p0);
            if (new_p0.x() < interval_t::zero && new_pr.x() > interval_t::zero) {
                auto s = segment_t{new_p0, new_pr};
                auto b = segment_t{point_t{interval_t::zero, new_p0.y()}, point_t{interval_t::zero, new_pr.y()}};
                auto index_last_intersection = out_intersection.size() - 1;
                assert(::boost::geometry::intersects(s, b));
                ::boost::geometry::intersection(s, b, out_intersection);
                auto y = out_intersection[index_last_intersection + 1].y();

                if (y > new_p0.y()) {
                    y = new_p0.y();
                }

                if (y < new_pr.y()) {
                    y = new_pr.y();
                };

                upper.emplace_back(point_t{interval_t::zero, y});
            }

            lo_start += std::get<2>(sl) - (lower.data() + lo_start);
            lower[lo_start] = new_pr;
        }

        inline void cut(const lowerbound_t &l) {
            auto su = search_upper_intersection(l);
            auto sl = search_lower_intersection(l);

            auto new_pl = std::get<0>(su).up_intersection(segment_t{*std::get<1>(su), *std::get<2>(su)});
            auto new_p1 = std::get<0>(sl).lo_intersection(segment_t{*std::get<1>(sl), *std::get<2>(sl)});

            if (new_pl.x() > new_p1.x()) {
                ::boost::geometry::set<0>(new_pl, new_p1.x());
            }

            if (new_pl.y() < new_p1.y()) {
                ::boost::geometry::set<1>(new_pl, new_p1.y());
            }

            up_start += std::get<1>(su) - (upper.data() + up_start);
            upper[up_start] = new_pl;

            lower.resize(lower.size() - (&lower.back() - std::get<2>(sl)));
            lower.emplace_back(new_p1);

            if (new_pl.x() < interval_t::zero && new_p1.x() > interval_t::zero) {
                auto s = segment_t{new_pl, new_p1};
                auto b = segment_t{point_t{interval_t::zero, new_pl.y()}, point_t{interval_t::zero, new_p1.y()}};
                assert(::boost::geometry::intersects(s, b));
                ::boost::geometry::intersection(s, b, out_intersection);
                auto y = out_intersection.back().y();
                if (y > new_pl.y()) {
                    y = new_pl.y();
                }

                if (y < new_p1.y()) {
                    y = new_p1.y();
                }

                lower.emplace_back(point_t{interval_t::zero, y});
            }
        }

        inline bool cut_with_upper_bound(const upperbound_t &u) {
            const auto pl = ul();
            const auto pr = lr();
            segment_t d{pl, pr};

            if (u.right().second.y() >= d.second.y()) {
                return true;
            } else if (u.left().first.y() >= d.first.y() && u.right().second.y() < d.second.y()) {
                assert(u.intersects(d));
                cut(u);
                return true;
            } else {
                //assert(u.left().first.y() < d.first.y());
                return false;
            }
        }

        inline bool cut_with_lower_bound(const lowerbound_t &l) {
            const auto pl = ul();
            const auto pr = lr();
            segment_t d{pl, pr};

            if (l.left().first.y() <= d.first.y()) {
                return true;
            } else if (l.left().first.y() > d.first.y() && l.right().second.y() <= d.second.y()) {
                assert(l.intersects(d));
                cut(l);
                return true;
            } else {
                //assert(l.right().second.y() > d.second.y());
                return false;
            }
        }

        inline bool update(upperbound_eq_t &u1, lowerbound_eq_t &l1) {
            //assert(u1.sign() == 0 && l1.sign() == 1);
            if (empty()) {
                init = boundaries_eq_t{u1, l1};
                return true;
            } else if (is_init()) {

                auto [u0, l0] = init.value();

                auto pl = u0.intersection(l1);
                auto p1 = u0.intersection(u1);

                auto _p1 = l0.intersection(l1);
                auto pr = l0.intersection(u1);

                if (pl.x() < interval_t::zero && pr.x() > interval_t::zero) {
                    if (p1.x() > interval_t::zero) {
                        upper.push_back(pl);
                        upper.push_back(point_t{interval_t::zero, u0._q});
                        upper.push_back(p1);
                    } else if (p1.x() < interval_t::zero) {
                        upper.push_back(pl);
                        upper.push_back(p1);
                        upper.push_back(point_t{interval_t::zero, u1._q});
                    } else {
                        upper.push_back(pl);
                        upper.push_back(p1);
                    }
                } else {
                    upper.push_back(pl);
                    upper.push_back(p1);
                }

                if (pr.x() > interval_t::zero && pl.x() < interval_t::zero) {
                    if (_p1.x() < interval_t::zero) {
                        lower.push_back(pr);
                        lower.push_back(point_t{interval_t::zero, l0._q});
                        lower.push_back(_p1);
                    } else if (_p1.x() > interval_t::zero) {
                        lower.push_back(pr);
                        lower.push_back(_p1);
                        lower.push_back(point_t{interval_t::zero, l1._q});
                    } else {
                        lower.push_back(pr);
                        lower.push_back(_p1);
                    }
                } else {
                    lower.push_back(pr);
                    lower.push_back(_p1);
                }

                //assert(pl.x() <= p1.x() && p1.x() <= pr.x());
                //assert(pr.x() >= _p1.x());
                //assert(pl.y() >= p1.y() && p1.y() >= pr.y());
                //assert(pr.y() <= _p1.y());

                init = std::nullopt;
                return true;
            } else {

                auto fix_polygon = [&](const auto &pl, const auto &pr) {
                    if (lower.back().x() >= upper.back().x() && lower.back().y() > upper.back().y()) {
                        upper.clear();
                        lower.clear();
                        upper.emplace_back(pl);
                        lower.emplace_back(pr);
                        up_start = 0;
                        lo_start = 0;
                        return false;
                    }
                    return true;
                };

                auto pl = ul();
                auto pr = lr();
                auto u = u1.split(pl.x(), pr.x());
                bool intersect_upper_bound = cut_with_upper_bound(u);
                intersect_upper_bound = intersect_upper_bound && fix_polygon(pl, pr);

                bool intersect_lower_bound;
                if (intersect_upper_bound) {
                    auto l = l1.split(ul().x(), lr().x());
                    intersect_lower_bound = cut_with_lower_bound(l);
                    intersect_lower_bound = intersect_lower_bound && fix_polygon(pl, pr);
                }

                out_intersection.clear();
                return intersect_upper_bound && intersect_lower_bound;
            }
        }

        [[nodiscard]] constexpr bool empty() const {
            return upper.empty() && lower.empty() && !is_init();
        }

        inline void clear() {
            upper.clear();
            up_start = 0;
            lower.clear();
            lo_start = 0;
            out_intersection.clear();

            //edges.clear();
            init = std::nullopt;
        }

        inline point_t ul() const {
            return upper[up_start];
        }

        inline point_t lr() const {
            return lower[lo_start];
        }

        [[nodiscard]] inline bool is_init() const {
            return init.has_value();
        }

        [[nodiscard]] bool check_polygon() const {
            // check upper, we want that p0.first <= p1.first <= p2.first ... and that p0.second >= p1.second >= p1.second...
            auto p0 = upper[up_start];
            point_t p1;
            for (auto i = up_start + 1; i <= upper.size(); ++i) {
                if (i == upper.size()) p1 = lr();
                else p1 = upper[i];
                auto b0 = segment_t{p0, p1};
                if (p0.x() > p1.x()) return false;
                if (p0.y() < p1.y()) return false;
                p0 = p1;
            }

            if (lower.back().x() >= upper.back().x() && lower.back().y() > upper.back().y())
                return false;

            // check lower, we want that p0.first >= p1.first >= p2.first ... and that p0.second <= p1.second <= p1.second...
            //assert(p0 == lower[lo_start]);
            p0 = lower[lo_start];
            for (auto i = lo_start + 1; i <= lower.size(); ++i) {
                if (i == lower.size()) p1 = ul();
                else p1 = lower[i];
                if (p0.x() < p1.x()) return false;
                if (p0.y() > p0.y()) return false;
                p0 = p1;
            }
            return true;
        }
    };
}