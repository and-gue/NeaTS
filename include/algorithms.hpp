#pragma once

#include <tuple>
#include <type_traits>
#include <cstdint>
#include <sdsl/int_vector.hpp>
#include <sux/bits/SimpleSelect.hpp>
#include <sux/bits/SimpleSelectHalf.hpp>
#include <sux/bits/SimpleSelectZero.hpp>
#include <sux/bits/SimpleSelectZeroHalf.hpp>
#include <set>
#include <unordered_set>
#include <span>
#include <ranges>
#include <climits>
#include <experimental/simd>

/** Computes (bits_per_correction > 0 ? 2^(bits_per_correction-1) - 1 : 0) without the conditional operator. */
#define BPC_TO_EPSILON(bits_per_correction) (((1ul << (bits_per_correction)) + 1) / 2 - 1)
#define LOG2(x) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll(x) - 1))

/** Computes the smallest integral value not less than x / y, where x and y must be positive integers. */
#define CEIL_UINT_DIV(x, y) ((x) / (y) + ((x) % (y) != 0))

namespace pfa::algorithm {

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

    template<typename T>
    struct AlignedAllocator {
        using value_type = T;

        T *allocate(std::size_t n) {
            return static_cast<T *>(std::aligned_alloc(64, sizeof(T) * n));
        }

        void deallocate(T *p, std::size_t n) {
            std::free(p);
        }
    };

    template<typename TypeIn, typename TypeOut = int64_t>
    inline std::vector<TypeOut> _preprocess_data(const std::vector<TypeIn> &in_data, int64_t bpc = 0,
                                                 size_t max_size = std::numeric_limits<size_t>::max()) {
        auto processed_data = std::vector<TypeOut>(in_data.size());
        if constexpr (std::is_signed_v<TypeIn>) {
            TypeIn min_data = *std::min_element(in_data.begin(), in_data.end());
            min_data = min_data < 0 ? (min_data - 1) : -1;
            auto epsilon = (TypeIn) BPC_TO_EPSILON(bpc);
            std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                           [min_data, epsilon](TypeIn d) -> TypeOut { return TypeOut(d - (min_data - epsilon)); });
        } else {
            static_assert(
                    std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut>, //&& sizeof(TypeIn) == sizeof(TypeOut),
                    "TypeIn must be unsigned and TypeOut must be signed and of the same size");
            TypeIn max_data = *std::max_element(in_data.begin(), in_data.end());
            TypeOut min_data = -1;
            auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
            auto max_val = std::numeric_limits<TypeOut>::max() - epsilon - 1;
            if (max_data > max_val) {
                std::cerr << "Warning: Data values are too large for the output type" << std::endl;


                for (auto i = 0; i < in_data.size(); ++i) {
                    if (in_data[i] > max_val) {
                        std::cerr << "Warning: Data value at index " << i << " is too large for the output type"
                                  << std::endl;
                        processed_data[i] = -(min_data - epsilon);
                    } else {
                        processed_data[i] = (TypeOut) (in_data[i]) - (min_data - epsilon);
                    }
                }
            } else {
                std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                               [min_data, epsilon](TypeIn d) -> TypeOut {
                                   return TypeOut(d - (min_data - epsilon));
                               });
                //processed_data = std::vector<TypeOut>(in_data.begin(), in_data.end());
                //throw std::runtime_error("Not implemented yet");
            }
        }
        return processed_data;
    }

    namespace io {

        template<typename TypeIn, typename TypeOut>
        std::vector<TypeOut> read_data_binary(const std::string &filename, bool first_is_size = true,
                                              size_t max_size = std::numeric_limits<size_t>::max()) {
            try {
                auto openmode = std::ios::in | std::ios::binary;
                if (!first_is_size)
                    openmode |= std::ios::ate;

                std::fstream in(filename, openmode);
                in.exceptions(std::ios::failbit | std::ios::badbit);

                size_t size;
                if (first_is_size)
                    in.read((char *) &size, sizeof(size_t));
                else {
                    size = static_cast<size_t>(in.tellg() / sizeof(TypeIn));
                    in.seekg(0);
                }
                size = std::min(max_size, size);

                std::vector<TypeIn> data(size);
                in.read((char *) data.data(), size * sizeof(TypeIn));
                in.close();
                if constexpr (std::is_same<TypeIn, TypeOut>::value)
                    return data;

                return std::vector<TypeOut>(data.begin(), data.end());
            }
            catch (std::ios_base::failure &e) {
                std::cerr << e.what() << std::endl;
                std::cerr << std::strerror(errno) << std::endl;
                exit(1);
            }
        }

        template<typename TypeIn>
        inline std::vector<TypeIn> simd_preprocess(const std::string& fn, auto bpc, bool first_is_size) {
            namespace stdx = std::experimental;
            using simd_t = stdx::native_simd<TypeIn>;
            constexpr auto simd_width = simd_t::size();

            auto simd_min = [](auto &&ptr, auto n) {
                simd_t simd_w;
                typename simd_t::value_type min_val = 0;
                auto j{0};
                for (; j + simd_width <= n; j += simd_width) {
                    simd_w.copy_from(ptr + j, stdx::element_aligned);
                    auto _min = stdx::hmin(simd_w);
                    min_val = std::min(min_val, _min);
                }
                for (; j < n; ++j) {
                    min_val = std::min(min_val, ptr[j]);
                }
                return min_val - 1;
            };

            auto simd_preprocess = [](auto &&ptr, uint32_t n, TypeIn v) {
                simd_t simd_w;
                simd_t simd_eps{v};

                auto j{0};
                for (; j + simd_width <= n; j += simd_width) {
                    simd_w.copy_from(ptr + j, stdx::element_aligned);
                    simd_w -= simd_eps;
                    simd_w.copy_to(ptr + j, stdx::element_aligned);
                }
                for (; j < n; ++j) {
                    ptr[j] -= v;
                }
            };

            auto data_vec = pfa::algorithm::io::read_data_binary<int64_t, int64_t>(fn, first_is_size);
            auto epsilon = static_cast<TypeIn>(bpc);
            simd_preprocess(data_vec.data(), data_vec.size(), simd_min(data_vec.data(), data_vec.size()) - epsilon);
            return data_vec;
        }

        template<typename TypeIn, typename TypeOut>
        std::vector<TypeOut, AlignedAllocator<TypeOut>>
        read_vector_binary(const std::string &filename, bool first_is_size = true,
                           size_t max_size = std::numeric_limits<size_t>::max()) {
            try {
                auto openmode = std::ios::in | std::ios::binary;
                if (!first_is_size)
                    openmode |= std::ios::ate;

                std::fstream in(filename, openmode);
                in.exceptions(std::ios::failbit | std::ios::badbit);

                size_t size;
                if (first_is_size)
                    in.read((char *) &size, sizeof(size_t));
                else {
                    size = static_cast<size_t>(in.tellg() / sizeof(TypeIn));
                    in.seekg(0);
                }
                size = std::min(max_size, size);

                std::vector<TypeIn, AlignedAllocator<TypeIn>> data(size);
                in.read((char *) data.data(), size * sizeof(TypeIn));
                in.close();
                if constexpr (std::is_same<TypeIn, TypeOut>::value)
                    return data;

                return std::vector<TypeOut, AlignedAllocator<TypeOut>>(data.begin(), data.end());
            }
            catch (std::ios_base::failure &e) {
                std::cerr << e.what() << std::endl;
                std::cerr << std::strerror(errno) << std::endl;
                exit(1);
            }
        }

        template<typename TypeIn, typename TypeOut = int64_t>
        inline std::vector<TypeOut>
        preprocess_data(const std::string &filename, int64_t bpc = 0, bool first_is_size = true,
                        size_t max_size = std::numeric_limits<size_t>::max()) {
            auto in_data = read_data_binary<TypeIn, TypeIn>(filename, first_is_size, max_size);
            return _preprocess_data<TypeIn, TypeOut>(in_data, bpc);
        }

        template<typename TypeIn, typename TypeOut = int64_t>
        inline std::vector<TypeIn>
        postprocess_data(std::vector<TypeOut> in_data, int64_t bpc = 0, TypeIn min_data = -1) {
            static_assert(std::is_signed_v<TypeOut>);
            auto processed_data = std::vector<TypeIn>(in_data.size());
            if constexpr (std::is_signed_v<TypeIn>) {
                auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
                std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                               [min_data, epsilon](TypeOut d) -> TypeIn { return TypeIn(d + (min_data - epsilon)); });

            } else {
                static_assert(
                        std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut>, //&& sizeof(TypeIn) == sizeof(TypeOut),
                        "TypeIn must be unsigned and TypeOut must be signed and of the same size");
                auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
                std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                               [min_data, epsilon](TypeOut d) -> TypeIn { return TypeIn(d + (min_data - epsilon)); });
            }
            return processed_data;
        }

        template<int64_t bpc, typename TypeIn = uint64_t, typename TypeOut = int64_t>
        void check_data(const std::string &full_filename) {
            constexpr int64_t epsilon = BPC_TO_EPSILON(bpc);
            auto original_data = read_data_binary<TypeIn, TypeIn>(full_filename);
            auto processed_data = preprocess_data<TypeIn, TypeOut>(full_filename, true, bpc);
            auto data = postprocess_data<TypeIn, TypeOut>(processed_data, bpc);

            for (auto i = 0; i < original_data.size(); ++i) {
                if constexpr (std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut> &&
                              sizeof(TypeIn) == sizeof(TypeOut)) {
                    auto max_val = std::numeric_limits<TypeOut>::max() - epsilon - 1;
                    if (original_data[i] > max_val) {
                        if (data[i] != 0) {
                            std::cout << i << ": " << data[i] << "!=" << 0 << std::endl;
                            exit(1);
                        }
                    } else if (original_data[i] != data[i]) {
                        std::cout << i << ": " << original_data[i] << "!=" << data[i] << std::endl;
                        exit(1);
                    }
                } else {
                    if (original_data[i] != data[i]) {
                        std::cout << i << ": " << original_data[i] << "!=" << data[i] << std::endl;
                        exit(1);
                    }
                }
            }
        }

        class csv_row {
        public:
            auto operator[](size_t index) const {
                return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] - (m_data[index] + 1));
            }

            [[nodiscard]] std::size_t size() const {
                return m_data.size() - 1;
            }

            void readNextRow(std::istream &str) {
                std::getline(str, m_line);

                m_data.clear();
                m_data.emplace_back(-1);
                std::string::size_type pos = 0;
                while ((pos = m_line.find(delimiter, pos)) != std::string::npos) {
                    m_data.emplace_back(pos);
                    ++pos;
                }
                // This checks for a trailing comma with no data after it.
                pos = m_line.size();
                m_data.emplace_back(pos);
            }

        private:
            std::string m_line;
            std::vector<std::string::size_type> m_data;
            char delimiter = ',';

        public:
            csv_row() = default;

            explicit csv_row(char delimiter) : delimiter(delimiter) {}

            friend std::istream &operator>>(std::istream &str, csv_row &data) {
                data.readNextRow(str);
                return str;
            }
        };

        class CSVIterator {
        public:
            typedef std::input_iterator_tag iterator_category;
            typedef csv_row value_type;
            typedef std::size_t difference_type;
            typedef csv_row *pointer;
            typedef csv_row &reference;

            CSVIterator(std::istream &str) : m_str(str.good() ? &str : nullptr) { ++(*this); }

            CSVIterator() : m_str(nullptr) {}

            // Pre Increment
            CSVIterator &operator++() {
                if (m_str) { if (!((*m_str) >> m_row)) { m_str = nullptr; }}
                return *this;
            }

            // Post increment
            CSVIterator operator++(int) {
                CSVIterator tmp(*this);
                ++(*this);
                return tmp;
            }

            csv_row const &operator*() const { return m_row; }

            csv_row const *operator->() const { return &m_row; }

            bool operator==(CSVIterator const &rhs) {
                return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));
            }

            bool operator!=(CSVIterator const &rhs) { return !((*this) == rhs); }

        private:
            std::istream *m_str;
            csv_row m_row{u8' '};
        };

    }
}

