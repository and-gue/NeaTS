#pragma once

#include "coefficients_space.hpp"
#include <memory>
#include <functional>

namespace neats {

    //ModelBuilderType<decltype(linear)> linear_model(ModelBuilderType<decltype(linear)>::ModelTypeId::Linear, linear);


    template<typename XType, typename YType, typename PolygonType, typename T1, typename T2, bool indexing = false>
    class PiecewiseOptimalModel {

        using ConvexPolygon = neats::internal::ConvexPolygon<PolygonType>;
        using UpperBound = typename ConvexPolygon::UpperBound;
        using LowerBound = typename ConvexPolygon::LowerBound;
        using Boundaries = typename ConvexPolygon::Boundaries;
        using Point = typename ConvexPolygon::Point;
        using Segment = typename ConvexPolygon::Segment;
        using DataType = std::pair<XType, YType>;

    public:
        enum class ModelTypeId : uint8_t {
            Constant,
            Linear,
            Exponential
        };

        /*
        using InternalLinearApplyType = std::function<XType(XType)>;
        using LinearApplyType = std::function<InternalLinearApplyType(T1, T2)>;

        template<class name>
        constexpr static LinearApplyType linear = [](T1 a, T2 b) -> InternalLinearApplyType {
            return InternalLinearApplyType{[a, b](XType x) -> XType {
                return static_cast<XType>(a * x + b);
            }};
        };
        */

        template<typename SlopeType, typename InterceptType>
        constexpr static auto linear(SlopeType a, InterceptType b) {
            return [a, b](XType x) -> long double {
                assert(a >= 0.0);
                return (a * static_cast<long double>(x) + b);
            };
        };

        using LinearApplyType = decltype(&linear<T1, T2>);
        using InternalLinearApplyType = decltype(linear<T1, T2>(std::declval<T1>(), std::declval<T2>()));

        constexpr static auto constant = [](T1 c) {
            return [c](XType x) {
                return static_cast<XType>(c);
            };
        };

        constexpr static auto exponential = [](T1 a, T2 b) {
            return [a, b](XType x) {
                return b * std::exp(a * x);
            };
        };

        template<typename ApplyFn>
        class ModelBuilderType {
            friend class PiecewiseOptimalModel;

        public:
            //ModelBuilderType(ModelBuilderType &&) = default;

            constexpr ModelBuilderType(ModelTypeId type, const ApplyFn &fn) : id{type},
                                                                              _fn(std::make_shared<ApplyFn>(fn)) {
                switch (id) {
                    case ModelTypeId::Linear:
                        _compute_bounds = std::make_shared<ComputeBoundsFn>(linear_bounds);
                        break;
                    case ModelTypeId::Exponential:
                        _compute_bounds = std::make_shared<ComputeBoundsFn>(exponential_bounds);
                        break;
                    case ModelTypeId::Constant:
                        _compute_bounds = std::make_shared<ComputeBoundsFn>(constant_bounds);
                        break;
                    default:
                        throw std::runtime_error("Unknown model type");
                }
            };

            template<class InterceptType = T1>
            struct Linear {
                XType key;
                //InternalLinearApplyType apply_fn;
                T2 slope;
                InterceptType intercept;

                //ModelTypeId linear_type;
                //std::shared_ptr<ApplyFn> apply_fn;

                Linear(XType k, T2 s, T1 i) : key{k}, slope{s} {
                    intercept = static_cast<InterceptType>(i); //(i < InterceptType{0}? -1.0 * i : i);
                }

                Linear(XType k, Segment diagonal) : key{k} {
                    auto p = diagonal.center();
                    //intercept = p.y();
                    slope = p.x();
                    intercept = p.y();

                    //intercept = static_cast<InterceptType>(p.y() < 0.0? -1.0 * p.y() : p.y());
                    //intercept += InterceptType{1};
                    //apply_fn = fn(p.x(), p.y());
                }

                template<class T>
                inline T get_intercept() const {
                    auto r = linear(slope, intercept)(key);
                    return r < T{0}? (-1 * r) : r;
                }
                /*
                template<class T>
                T get_intercept() const {
                    return static_cast<T>(intercept < T{0}? (-1 * intercept) : intercept);
                }
                */

                auto operator()(XType x) const {
                    return linear(slope, intercept)(x);
                }

                operator XType() { return key; };
            };

            template<typename LinearType>
            LinearType finish(const Segment& diagonal) {
                //auto [a, b] = diagonal.center();
                switch (id) {
                    case ModelTypeId::Linear:
                        return Linear{diagonal};
                        break;
                    case ModelTypeId::Exponential:
                        throw std::runtime_error("Exponential model not supported");
                    case ModelTypeId::Constant:
                        throw std::runtime_error("Constant model not supported");
                    default:
                        throw std::runtime_error("Unknown model type");
                }
            }

            //auto apply(auto x) { return _fn(x); }
            //using ApplyFnType = ApplyFn;



        private:

            static constexpr Boundaries linear_bounds(const DataType &p, const YType &eps) {
                // p.x = key, p.y = rank
                // a * p.x + b >= p.y - eps && a * p.x + b <= p.y + eps
                // b >= (p.y - eps) - a * p.x  && b <= (p.y + eps) - a * p.x
                // (p.y + eps) - a * p.x >= (p.y - eps) - a * p.x

                auto _eps = static_cast<int64_t>(eps);
                auto m = -static_cast<PolygonType>(p.first);
                auto uq = p.second + _eps;
                auto lq = p.second - _eps;

                LowerBound l{Segment(m, lq)};
                UpperBound u{Segment(m, uq)};
                return Boundaries{u, l};
            }

            static constexpr Boundaries exponential_bounds(const DataType &p, const YType &eps) {
                auto _eps = static_cast<int64_t>(eps);
                auto m = -static_cast<PolygonType>(p.first);
                auto uq = std::log(static_cast<PolygonType>(p.second + _eps));
                auto lq = std::log(static_cast<PolygonType>(p.second - _eps));

                LowerBound l{Segment{m, lq}};
                UpperBound u{Segment{m, uq}};
                return Boundaries{u, l};
            }

            static constexpr Boundaries constant_bounds(const DataType &p, const YType &eps) {
                throw std::runtime_error("Constant model not supported");
            }

            //using ComputeBoundsFn = Boundaries (*)(const DataType &, const YType &);
            using ComputeBoundsFn = std::function<Boundaries(DataType, YType)>;

        private:
            ModelTypeId id;
            std::shared_ptr<ApplyFn> _fn;
            std::shared_ptr<ComputeBoundsFn> _compute_bounds;
        };

        template<class KType, class Fn>
        struct Model {
            friend class PiecewiseOptimalModel;

            KType key;
            Segment _diagonal;
            ModelBuilderType<Fn> builder_type;

            explicit Model() = default;

            explicit Model(KType origin, Segment d, ModelBuilderType<Fn> type) : key(origin),
                                                                                 _diagonal(std::move(d)), builder_type(type) {}

            inline bool one_point() const {
                return _diagonal.p0() == _diagonal.p1();
            }

            operator KType() { return key; };

            template<typename Out>
            Out result() {
                return Out{key, _diagonal};//builder_type.template finish<Out>(_diagonal);
            }

        };

        PiecewiseOptimalModel() : epsilon{} {}

        PiecewiseOptimalModel(const YType eps) : epsilon(eps) {}

        inline Boundaries compute_bounds(const DataType &p) const {
            return model._compute_bounds->operator()(p, epsilon);
        }

        inline bool add_point(const DataType &p) {
            auto [l, u] = compute_bounds(p);
            return g.update(l, u);
        }

        using LinearModelType = Model<XType, LinearApplyType>;

        // In case indexing == true => (in_gen(i), *(value_begin + i)) else (y[i], in_gen(i))
        template<class It, class Fn>
        LinearModelType maximal_fragment(const Fn& transform_value, It value_begin, It value_end, uint32_t &offset) {
            DataType p0;
            bool intersect;
            auto n = std::distance(value_begin + offset, value_end);

            auto gen = [](auto&& i) -> auto {
                return i + 1;
            };

            auto first_value = transform_value(value_begin + offset);
            if (offset == 0) {
                if constexpr (indexing)
                    p0 = DataType{first_value, gen(offset)};
                else
                    p0 = DataType{gen(offset), first_value};

                add_point(p0);
            }

            for (uint32_t i = 1; i < n; ++i) {
                const auto value = transform_value(value_begin + offset + i);

                if constexpr (indexing)
                    p0 = DataType{value, gen(i + offset)};
                else
                    p0 = DataType{gen(i + offset), value};

                intersect = add_point(p0);

                if (!intersect) {
                    Segment diagonal = g.diagonal();
                    g.clear();
                    auto out = LinearModelType{first_value, diagonal, model};
                    offset += i;
                    p0 = DataType{value, gen(offset++)};
                    add_point(p0);
                    return out;
                }
            }

            offset += n;
            //const auto value = transform_value(value_begin + offset);
            Segment diagonal = g.diagonal();
            g.clear();
            auto out = LinearModelType{first_value, diagonal, model};
            return out;
        }

        template<class It>
        auto make_fragmentation(It begin, It end) {
            auto n = std::distance(begin, end);
            auto models = std::vector<LinearModelType>{};

            auto identity = [](auto i) -> auto {
                return *(i);
            };

            for (uint32_t offset = 0; offset < n;) {
                auto current_model = maximal_fragment(identity, begin, end, offset);
                models.emplace_back(current_model);
            }

            return models;
        }

        //using InternalLinearApplyType = std::function<T2(XType)>;
        //using LinearApplyType = std::function<InternalLinearApplyType(T1, T2)>;
        using LinearBuilderType = ModelBuilderType<LinearApplyType>;
    private:

        YType epsilon{};
        ConvexPolygon g{};
        LinearBuilderType model{ModelTypeId::Linear, linear<T1, T2>};
    };

    using PlaModel = PiecewiseOptimalModel<int64_t, uint32_t, double, float, double, true>;
    using Linear = PlaModel::LinearBuilderType::Linear<float>;

    template<typename PlaType, typename It>
    std::vector<typename PlaType::LinearModelType> MakeLinearFragmentation(auto epsilon,
                                                                           It begin, It end) {
        PlaType model{epsilon};
        return model.make_fragmentation(begin, end);
    }

    template<typename PlaType, typename It, typename Fn>
    std::vector<typename PlaType::LinearModelType> MakeLinearFragmentation(auto epsilon, Fn transform,
                                                                           It begin, It end) {
        PlaType m(epsilon);
        auto n = std::distance(begin, end);
        auto models = std::vector<typename PlaType::LinearModelType>{};

        for (uint32_t offset = 0; offset < n;) {
            auto current_model = m.maximal_fragment(transform, begin, end, offset);
            models.emplace_back(current_model);
        }

        /*
        auto linear_models = std::vector<typename PlaType::LinearBuilderType::Linear>{};
        linear_models.reserve(models.size());
        for (auto &&model: models) {
            Linear lm = model.template result<typename PlaType::LinearBuilderType::Linear>();
            linear_models.emplace_back(lm);
        }
        */

        return models;
    }
}