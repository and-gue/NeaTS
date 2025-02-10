#pragma once
#include <fstream>
#include <vector>
#include <numeric>
#include <iostream>
#include <cstring>

namespace fa::utils {

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

    template<int64_t bpc, typename TypeIn, typename TypeOut = int64_t>
    inline std::vector<TypeOut> preprocess_data(const std::string &filename, bool first_is_size = true,
                                         size_t max_size = std::numeric_limits<size_t>::max()) {
        auto in_data = read_data_binary<TypeIn, TypeIn>(filename, first_is_size, max_size);
        auto processed_data = std::vector<TypeOut>(in_data.size());
        if constexpr (std::is_signed_v<TypeIn>) {
            TypeIn min_data = *std::min_element(in_data.begin(), in_data.end());
            min_data = min_data < 0 ? (min_data - 1) : -1;
            auto epsilon = (TypeIn) BPC_TO_EPSILON(bpc);
            std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                           [min_data, epsilon](TypeIn d) -> TypeOut { return TypeOut(d - (min_data - epsilon)); });
        } else {
            static_assert(std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut>, //&& sizeof(TypeIn) == sizeof(TypeOut),
                          "TypeIn must be unsigned and TypeOut must be signed and of the same size");
            TypeIn max_data = *std::max_element(in_data.begin(), in_data.end());
            TypeOut min_data = -1;
            auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
            auto max_val = std::numeric_limits<TypeOut>::max() - epsilon - 1;
            if (max_data > max_val) {
                std::cerr << "Warning: Data values are too large for the output type" << std::endl;


                for (auto i = 0; i < in_data.size(); ++i) {
                    if (in_data[i] > max_val) {
                        std::cerr << "Warning: Data value at index " << i << " is too large for the output type" << std::endl;
                        processed_data[i] = - (min_data - epsilon);
                    } else {
                        processed_data[i] = (TypeOut)(in_data[i]) - (min_data - epsilon);
                    }
                }
            } else {
                std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                               [min_data, epsilon](TypeIn d) -> TypeOut { return TypeOut(d - (min_data - epsilon)); });
                //processed_data = std::vector<TypeOut>(in_data.begin(), in_data.end());
                //throw std::runtime_error("Not implemented yet");
            }
        }
        return processed_data;
    }

    template<int64_t bpc, typename TypeIn, typename TypeOut = int64_t>
    inline std::vector<TypeIn> postprocess_data(std::vector<TypeOut> in_data, TypeIn min_data = -1) {
        static_assert(std::is_signed_v<TypeOut>);
        auto processed_data = std::vector<TypeIn>(in_data.size());
        if constexpr (std::is_signed_v<TypeIn>) {
            auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
            std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                           [min_data, epsilon](TypeOut d) -> TypeIn { return TypeIn(d + (min_data - epsilon)); });

        } else {
            static_assert(std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut>, //&& sizeof(TypeIn) == sizeof(TypeOut),
                          "TypeIn must be unsigned and TypeOut must be signed and of the same size");
            auto epsilon = (TypeOut) BPC_TO_EPSILON(bpc);
            std::transform(in_data.begin(), in_data.end(), processed_data.begin(),
                           [min_data, epsilon](TypeOut d) -> TypeIn { return TypeIn(d + (min_data - epsilon)); });
        }
        return processed_data;
    }

    template<int64_t bpc, typename TypeIn = uint64_t, typename TypeOut = int64_t>
    void check_data(const std::string& full_filename) {
        constexpr int64_t epsilon = BPC_TO_EPSILON(bpc);
        auto original_data = fa::utils::read_data_binary<TypeIn, TypeIn>(full_filename);
        auto processed_data = fa::utils::preprocess_data<bpc, TypeIn, TypeOut>(full_filename);
        auto data = fa::utils::postprocess_data<bpc, TypeIn, TypeOut>(processed_data);

        for (auto i = 0; i < original_data.size(); ++i) {
            if constexpr (std::is_unsigned_v<TypeIn> && std::is_signed_v<TypeOut> && sizeof(TypeIn) == sizeof(TypeOut)) {
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





}