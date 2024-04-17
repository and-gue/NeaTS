#pragma once
#include <iostream>
#include <vector>
#include <sdsl/int_vector.hpp>
#include "float_pfa.hpp"

template<typename poly_t = double, typename x_t = uint32_t, typename y_t = int64_t, typename T1 = float, typename T2 = double>
class adaptive_approximation {

    using data_point = typename std::pair<x_t, y_t>;
    using polygon_t = typename std::pair<poly_t, poly_t>;

    enum fun_t {
        linear,
        exponential,
        quadratic
    };

    std::vector<x_t> starting_positions;
    std::vector<T1> coefficients_t0{};
    std::vector<T1> coefficients_t1{};
    std::vector<T2> coefficients_t2{};
    sdsl::int_vector<2> model_types{};

    int64_t error_bound;
    uint32_t _n;

    static inline bool intersection(polygon_t &g, const poly_t &l1, const poly_t &u1) {
        assert(l1 <= u1);
        auto &[l0, u0] = g;

        if (l1 > u0 || u1 < l0) {
            return false;
        }
        if (l1 > l0) l0 = l1;
        if (u1 < u0) u0 = u1;
        return true;
    }

    class linear_approximation {

        int64_t error_bound{};
        data_point p_start{};

    public:

        linear_approximation(int64_t e, const data_point &p) {
            error_bound = e;
            p_start = p;
        }

        std::pair<poly_t, poly_t> compute_bounds(const data_point &p_next) {
            auto l = poly_t{static_cast<poly_t>(p_next.second - error_bound - p_start.second) /
                           static_cast<poly_t>(p_next.first - p_start.first)};
            auto u = poly_t{static_cast<poly_t>(p_next.second + error_bound - p_start.second) /
                           static_cast<poly_t> (p_next.first - p_start.first)};
            return std::make_pair(l, u);
        }

        void update_starting_point(const data_point &p) {
            p_start = p;
        }

        x_t starting_point() {
            return p_start.first;
        }

        bool add_point(polygon_t &g, const data_point &p_next) {
            auto [l, u] = compute_bounds(p_next);
            return intersection(g, l, u);
        }

        struct linear {
            x_t start_pos;
            T1 slope;
            T2 intercept;

            y_t operator()(x_t x) {
                return ceil(slope * (x - start_pos) + intercept);
            }
        };

        linear create_fun(const polygon_t &g, x_t start_pos) {
            auto [l, u] = g;
            auto slope = (l + u) / 2.0;
            auto intercept = p_start.second;
            return linear{start_pos, static_cast<T1>(slope), static_cast<T2>(intercept)};
        }

    };

    class exponential_approximation {

        int64_t error_bound{};
        data_point p_start{};

    public:

        exponential_approximation(int64_t e, const data_point &p) {
            error_bound = e;
            p_start = p;
        }

        std::pair<poly_t, poly_t> compute_bounds(const data_point &p_next) {
            double l = (log(p_next.second - error_bound) - log(p_start.second)) / (p_next.first - p_start.first);
            double u = (log(p_next.second + error_bound) - log(p_start.second)) / (p_next.first - p_start.first);

            return std::make_pair(l, u);
        }

        void update_starting_point(const data_point &p) {
            p_start = p;
        }

        x_t starting_point() {
            return p_start.first;
        }

        bool add_point(polygon_t &g, const data_point &p_next) {
            auto [l, u] = compute_bounds(p_next);
            return intersection(g, l, u);
        }

        struct exponential {
            x_t start_pos;
            T1 a;
            T2 b;

            y_t operator()(x_t x) {
                return ceil(exp(a * (x - start_pos)) * b);
            }
        };

        exponential create_fun(const polygon_t &g, x_t start_pos) {
            auto [l, u] = g;
            auto a = (l + u) / 2.0;
            auto b = p_start.second;
            return exponential{start_pos, static_cast<T1>(a), static_cast<T2>(b)};
        }
    };

public:
    using pla_t = linear_approximation;
    using pea_t = exponential_approximation;
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t, poly_t, T1, T2>;
    using convex_polygon_t = poa_t::convex_polygon_t;
    using pqa_t = poa_t::pqa_t;

    template<typename It>
    adaptive_approximation(It first, It last, int64_t err) {

        error_bound = err;

        convex_polygon_t gq{};
        polygon_t gl{};
        polygon_t ge{};

        _n = std::distance(first, last);

        x_t start_pos = 0;
        x_t xl = 0;
        x_t xe = 0;
        x_t xq = 0;
        auto p_start_l = data_point{xl, *first};
        auto p_start_e = data_point{xe, *first};
        auto p_start_q = data_point{xq, *first};

        auto pla = pla_t{error_bound, p_start_l};
        auto pea = pea_t{error_bound, p_start_e};
        auto pqa = pqa_t{error_bound};

        std::vector<typename pla_t::linear> ll{};
        std::vector<typename pea_t::exponential> le{};
        std::vector<typename pqa_t::fun_t> lq{};

        std::vector<fun_t> fun_types;

        bool fcf, frv = false;
        double alfa = 0.5;

        auto advance_model = [&](fun_t ft, const auto &p_next, const auto& p_start) {
            switch (ft) {
                case fun_t::linear:
                    if (!pla.add_point(gl, p_next)) {
                        auto f = pla.create_fun(gl, start_pos - xl);
                        ll.push_back(f);
                        xl = 0;
                        pla.update_starting_point(data_point{xl, p_next.second});
                        gl = polygon_t{};
                    }
                    break;
                case fun_t::exponential:
                    if (!pea.add_point(ge, p_next)) {
                        auto f = pea.create_fun(ge, start_pos - xe);
                        le.push_back(f);
                        xe = 0;
                        pea.update_starting_point(data_point{xe, p_next.second});
                        ge = polygon_t{};
                    }
                    break;
                case fun_t::quadratic:
                    if (!pqa.add_point(gq, p_start_q, p_next)) {
                        auto f = pqa.create_fun(gq, data_point{start_pos - xq, p_start_q.second});
                        lq.push_back(f);
                        xq = 0;
                        p_start_q = data_point{xq, p_next.second};
                        gq.clear();
                    }
                    break;
            }
        };

        fun_t bf;
        double npl, npe, npq;
        data_point p_next_l, p_next_e, p_next_q;

        while (start_pos < (_n - 1)) {

            if (!fcf) {
                ++start_pos;
                p_next_l = data_point{++xl, *(first + start_pos)};
                p_next_e = data_point{++xe, *(first + start_pos)};
                p_next_q = data_point{++xq, *(first + start_pos)};

                advance_model(fun_t::linear, p_next_l, p_start_l);
                advance_model(fun_t::exponential, p_next_e, p_start_e);
                advance_model(fun_t::quadratic, p_next_q, p_start_q);

                if (!ll.empty() && !le.empty() && !lq.empty()) {

                    auto spl = ll.back().start_pos;
                    auto spe = le.back().start_pos;
                    auto spq = lq.back().get_start();

                    npl = ll.size() * 2 + (spl == start_pos ? 2 : 2 * alfa);
                    npe = le.size() * 2 + (spe == start_pos ? 2 : 2 * alfa);
                    npq = lq.size() * 3 + (spq == start_pos ? 3 : 3 * alfa);

                    if (npl <= npe && npl <= npq) {
                        bf = fun_t::linear;
                        if (spl != start_pos) fcf = true;
                        else {
                            frv = true;
                            for (auto &&f: ll) {
                                coefficients_t1.push_back(f.slope);
                                coefficients_t2.push_back(f.intercept);
                                starting_positions.push_back(f.start_pos);
                                fun_types.push_back(fun_t::linear);
                            }
                        }
                    } else if (npe <= npl && npe <= npq) {
                        bf = fun_t::exponential;
                        if (spe != start_pos) fcf = true;
                        else {
                            frv = true;
                            for (auto &&f: le) {
                                coefficients_t1.push_back(f.a);
                                coefficients_t2.push_back(f.b);
                                starting_positions.push_back(f.start_pos);
                                fun_types.push_back(fun_t::exponential);
                            }
                        }
                    } else {
                        bf = fun_t::quadratic;
                        if (spq != start_pos) fcf = true;
                        else {
                            frv = true;
                            for (auto &&f: lq) {
                                auto t = f.parameters();
                                coefficients_t0.push_back(std::get<1>(t).value());
                                coefficients_t1.push_back(std::get<2>(t));
                                coefficients_t2.push_back(std::get<3>(t));
                                starting_positions.push_back(f.get_start());
                                fun_types.push_back(fun_t::quadratic);
                            }
                        }
                    }
                }
            } else {

                switch (bf) {
                    case fun_t::linear:
                        p_next_l = data_point{++xl, *(first + ++start_pos)};
                        advance_model(bf, p_next_l, p_start_l);

                        if (xl == 0) {
                            for (auto &&f: ll) {
                                coefficients_t1.push_back(f.slope);
                                coefficients_t2.push_back(f.intercept);
                                starting_positions.push_back(f.start_pos);
                                fun_types.push_back(fun_t::linear);
                                fcf = false;
                                frv = true;
                            }
                        }
                        break;

                    case fun_t::exponential:
                        p_next_e = data_point{++xe, *(first + ++start_pos)};
                        advance_model(bf, p_next_e, p_start_e);

                        if (xe == 0) {
                            for (auto &&f: le) {
                                coefficients_t1.push_back(f.a);
                                coefficients_t2.push_back(f.b);
                                starting_positions.push_back(f.start_pos);
                                fun_types.push_back(fun_t::exponential);
                                fcf = false;
                                frv = true;
                            }
                        }

                        break;

                    case fun_t::quadratic:
                        p_next_q = data_point{++xq, *(first + ++start_pos)};
                        advance_model(bf, p_next_q, p_start_q);

                        if (xq == 0) {
                            for (auto &&f: lq) {
                                auto t = f.parameters();
                                coefficients_t0.push_back(std::get<1>(t).value());
                                coefficients_t1.push_back(std::get<2>(t));
                                coefficients_t2.push_back(std::get<3>(t));
                                starting_positions.push_back(f.get_start());
                                fun_types.push_back(fun_t::quadratic);
                                fcf = false;
                                frv = true;
                            }
                            break;
                        }
                }

                if (frv) {
                    ll.clear();
                    le.clear();
                    lq.clear();
                    frv = false;
                    npl = npe = npq = 0;

                    xl = xe = xq = 0;
                    p_start_l = data_point{xl, *(first + start_pos)};
                    p_start_e = data_point{xe, *(first + start_pos)};
                    p_start_q = data_point{xq, *(first + start_pos)};
                    pla.update_starting_point(p_start_l);
                    pea.update_starting_point(p_start_e);
                    //pqa.update_starting_point(p_start_q);

                    gq.clear();
                    gl = polygon_t{};
                    ge = polygon_t{};
                }
            }
        }

        if (!fcf) {
            npl = (ll.size() + 1) * 2;
            npe = (le.size() + 1) * 2;
            npq = (lq.size() + 1) * 3;
            if (npl <= npe && npl <= npq) {
                bf = fun_t::linear;
            } else if (npe <= npl && npe <= npq) {
                bf = fun_t::exponential;
            } else {
                bf = fun_t::quadratic;
            }
        }

        switch (bf) {
            case fun_t::linear: {
                auto fl = pla.create_fun(gl, start_pos - xl);
                ll.push_back(fl);
                for (auto &&f: ll) {
                    coefficients_t1.push_back(f.slope);
                    coefficients_t2.push_back(f.intercept);
                    starting_positions.push_back(f.start_pos);
                    fun_types.push_back(fun_t::linear);
                }
                break;
            }
            case fun_t::exponential: {
                auto fe = pea.create_fun(ge, start_pos - xe);
                le.push_back(fe);
                for (auto &&f: le) {
                    coefficients_t1.push_back(f.a);
                    coefficients_t2.push_back(f.b);
                    starting_positions.push_back(f.start_pos);
                    fun_types.push_back(fun_t::exponential);
                }
                break;
            }
            case fun_t::quadratic: {
                auto fq = pqa.create_fun(gq, data_point{start_pos - xq, p_start_q.second});
                lq.push_back(fq);
                for (auto &&f: lq) {
                    auto t = f.parameters();
                    coefficients_t0.push_back(std::get<1>(t).value());
                    coefficients_t1.push_back(std::get<2>(t));
                    coefficients_t2.push_back(std::get<3>(t));
                    starting_positions.push_back(f.get_start());
                    fun_types.push_back(fun_t::quadratic);
                }
                break;
            }
        }

        model_types = sdsl::int_vector<2>(fun_types.size());
        for (auto i = 0; i < fun_types.size(); ++i) {
            model_types[i] = static_cast<uint8_t>(fun_types[i]);
        }
    }

    [[nodiscard]] size_t size_in_bits() const {
        return starting_positions.size() * sizeof(uint32_t) * 8 +
                coefficients_t0.size() * sizeof(T1) * 8 +
                coefficients_t1.size() * sizeof(T1) * 8 +
                coefficients_t2.size() * sizeof(T2) * 8 +
                model_types.size() * 2;
    }

    std::vector<y_t> decompress() const {
        std::vector<y_t> out(_n);

        auto num_partitions = this->starting_positions.size();

        auto offset_coefficients_t0 = 0;
        auto offset_coefficients = 0;

        uint32_t i_segment = 0;
        uint32_t start_segment = 0;
        uint32_t end_segment;
        x_t i = 0;

        while (i < _n) {
            if (i_segment >= (num_partitions - 1)) end_segment = _n;
            else end_segment = starting_positions[i_segment + 1];

            fun_t t = static_cast<fun_t>(model_types[i_segment]);
            switch (t) {
                case fun_t::linear: {
                    auto slope = coefficients_t1[offset_coefficients];
                    auto intercept = coefficients_t2[offset_coefficients];
                    offset_coefficients++;
                    auto start_pos = starting_positions[i_segment];
                    for (uint32_t j = start_segment; j < end_segment; ++j) {
                        out[i++] = ceil(slope * (j + 1 - start_pos) + intercept);
                    }
                    break;
                }

                case fun_t::quadratic: {
                    const auto a = coefficients_t0[offset_coefficients_t0++];
                    const auto b = coefficients_t1[offset_coefficients];
                    const auto c = coefficients_t2[offset_coefficients];
                    offset_coefficients++;
                    auto start_pos = starting_positions[i_segment];
                    for (uint32_t j = start_segment; j < end_segment; ++j) {
                        out[i++] = ceil(a * (j - start_pos) * (j - start_pos) + b * (j - start_pos) + c);
                    }
                    break;
                }

                case fun_t::exponential: {
                    const auto a = coefficients_t1[offset_coefficients];
                    const auto b = coefficients_t2[offset_coefficients];
                    offset_coefficients++;
                    auto start_pos = starting_positions[i_segment];
                    for (uint32_t j = start_segment; j < end_segment; ++j) {
                        out[i++] = ceil(exp(a * ((j - start_pos) + 1)) * b);
                    }
                    break;
                }
            }
            ++i_segment;
            start_segment = end_segment;
        }

        return out;
    }

    template<typename It>
    bool check_partitions(It first, It last) {
        const auto decompressed = this->decompress();
        //int64_t epsilon = BPC_TO_EPSILON(this->bpc);

        if (std::distance(first, last) != _n) {
            std::cerr << "Error: input size is different from the size of the compressed data" << std::endl;
            return false;
        }

        for (auto i = 0; i < _n; ++i) {
            int64_t epsilon = error_bound;
            auto error = *(first + i) - decompressed[i];
            if (error > epsilon || error < -(epsilon + 1)) {
                std::cerr << "Error at position [" << i << "] is " << error << std::endl;
                //return false;
            }
        }
        return true;
    }

};