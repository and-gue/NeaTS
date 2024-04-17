#include "../lib/sdsl-lite/external/googletest/googletest/include/gtest/gtest.h"
#include "../include/float_coefficient_space.hpp"
#include "../include/float_pfa.hpp"
#include "../include/utils.hpp"
#include "../include/NeaTSL.hpp"
#include "../include/NeaTS.hpp"
#include <random>
#include <numeric>

#include <filesystem>

using x_t = std::uint64_t;
using y_t = int64_t;
using data_point = typename std::pair<x_t, y_t>;

std::pair<double, double> min_max(double x, uint8_t bpc) {

    uint64_t t = (1 << bpc) - 1ULL;
    // Set the two least significant bits to 00
    uint64_t y0_bits = *reinterpret_cast<uint64_t *>(&x) & ~t;

    // Set the two least significant bits to 11
    uint64_t y1_bits = *reinterpret_cast<uint64_t *>(&x) | t;

    // Convert the bit patterns back to doubles
    double y0 = *reinterpret_cast<double *>(&y0_bits);
    double y1 = *reinterpret_cast<double *>(&y1_bits);

    return std::make_pair(y0, y1);
}

/* Given a directory path returns a vector of all the files in the directory */
std::vector<std::string> get_files(const std::string &path) {
    std::vector<std::string> files;
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        //put only files with the extension .bin
        if (entry.path().extension() == ".bin")
            files.push_back(entry.path());
    }
    return files;
}

template<typename A>
inline void test_approximation(const auto& fn, auto first_size = true) {
    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn, first_size);
    //data = std::vector<int64_t>(data.begin() + 9335500, data.begin() + 9335500 + 200);
    //std::cout << data.size() << std::endl;
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= (min_data - BPC_TO_EPSILON(20)); });
    auto bpc = 0;
    while (bpc != 20) {
        int64_t epsilon = BPC_TO_EPSILON(bpc);
        auto pa = A{epsilon};
        auto res = pa.make_approximation(data.begin(), data.end());
        auto approx = pa.get_approximations(res, data.size());

        int64_t num_errors = 0;
        int64_t max_error = 0;
        for (auto i = 0; i < data.size(); ++i) {
            auto err = static_cast<int64_t>(data[i] - approx[i]);
            if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
                ++num_errors;
                auto abs_err = std::abs(err - epsilon);
                max_error = abs_err > max_error ? abs_err : max_error;
                std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
            }
        }

        std::cout << "Dataset: " << fn << ", size dataset: " << data.size()
                  << ", epsilon: " << epsilon
                  << ", num errors: " << num_errors
                  << ", max error: " << max_error
                  << ", num functions: " << res.size() << std::endl;

        if (bpc == 0) bpc += 2;
        else ++bpc;
    }
}

TEST(pla, app) {
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
    for (const auto &fn : fns) {
        test_approximation<poa_t::pla_t>(fn, true);
    }
}

TEST(pqa, app) {
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
    for (const auto &fn : fns) {
        test_approximation<poa_t::pqa_t>(fn, true);
    }
}

TEST(psa, app) {
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
    for (const auto &fn : fns) {
        test_approximation<poa_t::psa_t>(fn, true);
    }
}

TEST(pea, app) {
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
    for (const auto &fn : fns) {
        test_approximation<poa_t::pea_t>(fn, true);
    }
}

TEST(poa, app) {
    auto fn = "/data/citypost/neat_datasets/binary/big/geolife-lon.bin";
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t, long double, double, double>;
    using A = poa_t::psa_t;
    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn, true);
    //data = std::vector<int64_t>(data.begin() + 9335500, data.begin() + 9335500 + 200);
    //std::cout << data.size() << std::endl;
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= (min_data - BPC_TO_EPSILON(21)); });

    auto bpc = 0;
   while (bpc != 20) {
    int64_t epsilon = BPC_TO_EPSILON(bpc);
    auto pa = A{epsilon};
    auto res = pa.make_approximation(data.begin(), data.end());
    auto approx = pa.get_approximations(res, data.size());

    int64_t num_errors = 0;
    int64_t max_error = 0;
    for (auto i = 0; i < data.size(); ++i) {
        auto err = static_cast<int64_t>(data[i] - approx[i]);
        if (epsilon == 0 && err == -1) std::cerr << "ERROR: " << i << std::endl;
        if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
            ++num_errors;
            auto abs_err = std::abs(err - epsilon);
            max_error = abs_err > max_error ? abs_err : max_error;
            std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
        }
    }

    std::cout << "Dataset: " << fn << ", size dataset: " << data.size()
              << ", epsilon: " << epsilon
              << ", num errors: " << num_errors
              << ", max error: " << max_error
              << ", num functions: " << res.size() << std::endl;

    if (bpc == 0) bpc += 2;
    else ++bpc;
    }
}

TEST(neats, approximation) {
    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    neats::lossy_compressor<x_t, y_t, BPC_TO_EPSILON(17)> lc;
    auto fn = "./data/citypost/neat_datasets/binary/big/dust.bin";
    //test_approximation<poa_t::pla_t>(fn, true);
//
    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
//
    auto epsilon = lc.epsilon();
    std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
    lc.partitioning(data.begin(), data.end());
//
    std::decay<decltype(data)>::type decompressed(data.size());
    lc.decompress(std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.begin()),
                  std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.end()));
//
    int64_t num_errors = 0;
    int64_t max_error = 0;
    for (auto i = 0; i < data.size(); ++i) {
        auto err = static_cast<int64_t>(data[i] - decompressed[i]);
        if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
            ++num_errors;
            auto abs_err = std::abs(err - epsilon);
            max_error = abs_err > max_error ? abs_err : max_error;
            //std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
        }
    }
//
    std::cout << "dataset: " << fn
              << ", size: " << data.size()
              << ", epsilon: " << epsilon
              << ", #errors: " << num_errors
              << ", max_error: " << max_error << std::endl;
//
    lc.print_details();
}

TEST(neats_lossless, approximation) {

    //filename, delta
    //dust, 13.872

    //using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
    constexpr auto bpc = 17;
    constexpr int64_t epsilon = BPC_TO_EPSILON(bpc);
    neats::lossless_compressor<x_t, y_t, bpc, double, float, double, false> lc;
    auto fn = "/data/citypost/neat_datasets/binary/big/dust.bin";
    //test_approximation<poa_t::psa_t>(fn, true);
//
    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
    auto min_data = *std::min_element(data.begin(), data.end());
    auto max_data = *std::max_element(data.begin(), data.end());
    std::cout << LOG2(max_data - min_data) << std::endl;
    min_data = min_data < 0 ? (min_data - 1) : -1;
//

    data = std::vector<int64_t>(data.begin(), data.end());
    std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
    lc.partitioning(data.begin(), data.end());
//
    std::decay<decltype(data)>::type decompressed(data.size());
    lc.decompress(std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.begin()),
                  std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.end()));

    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
            //exit(1);
        }
        EXPECT_EQ(data[i], decompressed[i]);
    }

//
    lc.size_info();
    lc.to_csv("neats_dust.csv");

    std::cout << "BIT SIZE: " << lc.size_in_bits() << ", BYTE SIZE: " << ceil(lc.size_in_bits() / 8.0)
              << ", COMPRESSION RATIO: " << (long double)(lc.size_in_bits()) / (long double)(data.size() * sizeof(y_t) * 8) << std::endl;
//
}

//TEST(pla_boost, approximation) {
//    using poa_t = fa::pfa::piecewise_optimal_approximation<x_t, y_t>;
//
//    auto fns = get_files("../data/df/");
//    //auto fns = std::vector<std::string>{"../data/df/geolife-lon.bin"};
//
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//        auto min_data = *std::min_element(data.begin(), data.end());
//        min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 0;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//            auto pla = poa_t::pla_t{epsilon};
//            auto res = pla.make_approximation(data.begin(), data.end());
//
//            auto approx = pla.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1)) || (epsilon == 0 && err != 0)) {
//                    std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                }
//            }
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PLA functions: " << num_linear
//                      << ", #errors: " << num_errors
//                      << ", max_error: " << max_error << std::endl;
//            ++bpc;
//            break;
//        }
//    }
//}
//
//TEST(pea_boost, approximation) {
//
//    using poa_t = fa::boost::pfa::piecewise_optimal_approximation<x_t, y_t, long double, float, double>;
//
//    //auto fns = get_files("../data/df/");
//
//    auto fns = std::vector<std::string>{"../data/df/dust.bin"};
//    //auto fn = "../data/big/geolife-lon.bin";
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//
//        data = std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 8, 10};
//
//        //auto min_data = *std::min_element(data.begin(), data.end());
//        //min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 0;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            //std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//
//            //data.erase(data.begin(), data.begin() + 91);
//            //data.erase(data.begin() + 3, data.end());
//
//            auto pea = poa_t::pla_t{epsilon};
//            auto res = pea.make_approximation(data.begin(), data.end());
//
//            auto approx = pea.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//                    //std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                    //std::cout << "ERROR: " << max_error << std::endl;
//                    //exit(1);
//                }
//            }
//
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PEA functions: " << num_linear
//                      << ", #errors: " << num_errors
//                      << ", max_error: " << max_error << std::endl;
//            ++bpc;
//            break;
//        }
//    }
//}
//
//TEST(psa_boost, approximation) {
//    using poa_t = fa::boost::pfa::piecewise_optimal_approximation<x_t, y_t>;
//
//    //auto fns = std::vector<std::string>{"/data/citypost/neat_datasets/binary/big/uk.bin"};
//
//    auto fns = get_files("../data/df/");
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//        auto min_data = *std::min_element(data.begin(), data.end());
//        min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 0;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//            auto psa = poa_t::psa_t{epsilon};
//            auto res = psa.make_approximation(data.begin(), data.end());
//
//            auto approx = psa.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                    std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                }
//            }
//
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PSA functions: " << num_linear
//                      << ", #errors: " << num_errors
//                      << ", max_error: " << max_error << std::endl;
//            ++bpc;
//            break;
//        }
//    }
//}
//
//TEST(ppa_boost, approximation) {
//    using poa_t = fa::boost::pfa::piecewise_optimal_approximation<x_t, y_t>;
//
//    auto fns = get_files("../data/df/");
//    //auto fns = std::vector<std::string>{"/data/citypost/neat_datasets/binary/big/uk.bin"};
//
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//        auto min_data = *std::min_element(data.begin(), data.end());
//        min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 0;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//            auto ppa = poa_t::ppa_t{epsilon};
//            auto res = ppa.make_approximation(data.begin(), data.end());
//
//            auto approx = ppa.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                    std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                }
//            }
//
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PPA functions: " << num_linear
//                      << ", #errors: " << num_errors
//                      << ", max_error: " << max_error << std::endl;
//            ++bpc;
//            break;
//        }
//    }
//}
//
//TEST(plg_boost, approximation) {
//    using poa_t = fa::boost::pfa::piecewise_optimal_approximation<x_t, y_t>;
//
//    auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
//    //auto fns = std::vector<std::string>{"/data/citypost/neat_datasets/binary/big/dust.bin"};
//
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//        auto min_data = *std::min_element(data.begin(), data.end());
//        min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 2;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//            auto plg = poa_t::plg_t{epsilon};
//            auto res = plg.make_approximation(data.begin(), data.end());
//
//            auto approx = plg.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                    std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                }
//            }
//
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PLG functions: " << num_linear
//                      << ", #errors: " << num_errors
//                      << ", max_error: " << max_error << std::endl;
//            ++bpc;
//        }
//    }
//}
//
//TEST(pqa, approximation) {
//    using poa_t = fa::boost::pfa::piecewise_optimal_approximation<x_t, y_t, long double, double, double>;
//
//    //auto fns = get_files("/data/citypost/neat_datasets/binary/big/");
//    auto fns = std::vector<std::string>{"../data/df/uk.bin"};
//
//    for (const auto &fn: fns) {
//        auto num_linear = 0;
//        auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//        auto min_data = *std::min_element(data.begin(), data.end());
//        min_data = min_data < 0 ? (min_data - 1) : -1;
//
//        auto bpc = 5;
//        while (num_linear != 1) {
//            int64_t epsilon = BPC_TO_EPSILON(bpc);
//            std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//            auto pqa = poa_t::pqa_t{epsilon};
//            auto res = pqa.make_approximation(data.begin(), data.end());
//            auto approx = pqa.get_approximations(res, data.size());
//
//            int64_t num_errors = 0;
//            int64_t max_error = 0;
//            for (auto i = 0; i < data.size(); ++i) {
//                auto err = static_cast<int64_t>(data[i] - approx[i]);
//                if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//                    ++num_errors;
//                    auto abs_err = std::abs(err - epsilon);
//                    max_error = abs_err > max_error ? abs_err : max_error;
//                    std::cout << i << ": " << data[i] << "!=" << approx[i] << std::endl;
//                }
//            }
//
//            EXPECT_EQ(max_error, 0);
//
//            num_linear = res.size();
//            std::cout << "dataset: " << fn
//                      << ", size: " << data.size()
//                      << ", epsilon: " << epsilon
//                      << ", num PQA functions: " << num_linear << std::endl;
//                      //<< ", #errors: " << num_errors
//                      //<< ", max_error: " << max_error << std::endl;
//            bpc += (bpc == 0) + 1;
//            break;
//        }
//    }
//}
//
//TEST(neats, approximation) {
//    neats::lossy_compressor<x_t, y_t, BPC_TO_EPSILON(17)> lc;
//    auto fn = "../data/big/dust.bin";
//
//    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//    //auto min_data = *std::min_element(data.begin(), data.end());
//    //min_data = min_data < 0 ? (min_data - 1) : -1;
//
//    auto epsilon = lc.epsilon();
//    //std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//    lc.partitioning(data.begin(), data.end());
//
//    std::decay<decltype(data)>::type decompressed(data.size());
//    lc.decompress(std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.begin()),
//                  std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.end()));
//
//    int64_t num_errors = 0;
//    int64_t max_error = 0;
//    for (auto i = 0; i < data.size(); ++i) {
//        auto err = static_cast<int64_t>(data[i] - decompressed[i]);
//        if ((err > 0 && err > epsilon) || (err < 0 && err < (-epsilon - 1))) {
//            ++num_errors;
//            auto abs_err = std::abs(err - epsilon);
//            max_error = abs_err > max_error ? abs_err : max_error;
//            //std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
//        }
//    }
//
//    std::cout << "dataset: " << fn
//              << ", size: " << data.size()
//              << ", epsilon: " << epsilon
//              << ", #errors: " << num_errors
//              << ", max_error: " << max_error << std::endl;
//
//    //lc.print_details();
//}
//
//TEST(neats, lossless) {
//    neats::lossless_compressor<x_t, y_t, 23> lc;
//    auto fn = "../data/big/dust.bin";
//
//    auto data = fa::utils::read_data_binary<int64_t, int64_t>(fn);
//    auto min_data = *std::min_element(data.begin(), data.end());
//    min_data = min_data < 0 ? (min_data - 1) : -1;
//
//    int64_t epsilon = BPC_TO_EPSILON(24);
//    std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });
//    lc.partitioning(data.begin(), data.end());
//
//    std::decay<decltype(data)>::type decompressed(data.size());
//
//    lc.decompress(std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.begin()),
//                  std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.end()));
//
//    for (auto i = 0; i < data.size(); ++i) {
//        if (data[i] != decompressed[i]) {
//            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
//            exit(1);
//        }
//        EXPECT_EQ(data[i], decompressed[i]);
//    }
//
//    std::cout << "BIT SIZE: " << lc.size_in_bits() << ", BYTE SIZE: " << std::ceil(lc.size_in_bits() / 8.0)
//              << ", COMPRESSION RATIO: " << (lc.size_in_bits() / 8.0) / (data.size() * sizeof(y_t)) << std::endl;
//
//
//    //lc.print_info(
//    //        data.begin(), data.end(),
//    //        std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.begin()), std::forward<std::decay<decltype(data)>::type::iterator>(decompressed.end()));
//
//}