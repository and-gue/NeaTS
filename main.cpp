#include <iostream>
#include "NeaTS.hpp"
#include "utils.hpp"
#include <map>
#include <getopt.h>

template<typename x_t = uint32_t, typename y_t = int64_t, typename poly = double, typename T1 = float, typename T2 = double, int64_t... bpc>
constexpr auto init_neats(std::integer_sequence<int64_t, bpc...>) {
    using neats_variant = std::variant<typename neats::lossless_compressor<x_t, y_t, bpc + 2, poly, T1, T2>...>;
    std::map<int64_t, neats_variant> neats_map = {{bpc + 2, typename neats::lossless_compressor<x_t, y_t, bpc + 2, poly, T1, T2>()}...};
    return neats_map;
}

template<typename x_t = uint32_t, typename y_t = int64_t, typename poly = double, typename T1 = float, typename T2 = double, int64_t... bpc>
constexpr auto init_lavector(std::integer_sequence<int64_t, bpc...>) {
    using la_variant = std::variant<typename neats::lossless_compressor<x_t, y_t, bpc + 2, poly, T1, T2, true>...>;
    std::map<int64_t, la_variant> la_map = {{bpc + 2, typename neats::lossless_compressor<x_t, y_t, bpc + 2, poly, T1, T2, true>()}...};
    return la_map;
}


template<typename x_t = uint32_t, typename y_t = int64_t, typename poly = double, typename T1 = float, typename T2 = double>
void inline run(const std::string& fn, uint8_t bpc = 0) {
    // take the base name of fn (without its extension) and use it as the name of the dataset
    std::string ds_name = fn.substr(fn.find_last_of("/\\") + 1);
    ds_name = ds_name.substr(0, ds_name.find_last_of("."));

    std::string neats_fn = "neats_" + ds_name + ".csv";
    std::string la_fn = "la_" + ds_name + ".csv";

    assert(bpc <= 40 && "bpc must be less than or equal to 40");

    auto neats_map = init_neats<x_t, y_t, poly, T1, T2>(std::make_integer_sequence<int64_t, 40>{});

    auto data = fa::utils::read_data_binary<y_t, y_t>(fn);

    auto min_data = *std::min_element(data.begin(), data.end());
    auto max_data = *std::max_element(data.begin(), data.end());

    bpc = bpc == 0? LOG2(max_data - min_data) : bpc;
    min_data = min_data < 0 ? (min_data - 1) : -1;

    int64_t epsilon = BPC_TO_EPSILON(bpc);
    std::for_each(data.begin(), data.end(), [min_data, epsilon](auto &d) { d -= (min_data - epsilon); });

    auto compressor = neats_map[bpc];
    //neats::lossless_compressor<x_t, y_t, 17, float, double> nea;
    std::visit([&](auto &&neats_compressor) {
        neats_compressor.partitioning(data.begin(), data.end());
    }, compressor);

    typename std::decay<decltype(data)>::type decompressed(data.size());
    std::visit([&](auto &&neats_compressor) {
        neats_compressor.decompress(std::forward<typename std::decay<decltype(data)>::type::iterator>(decompressed.begin()),
                                    std::forward<typename std::decay<decltype(data)>::type::iterator>(decompressed.end()));
    }, compressor);

    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
            //exit(1);
        }
    }

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    auto compressed_bit_size = std::visit([](auto &&neats_compressor) {
        return neats_compressor.size_in_bits();
    }, compressor);


    std::cout << "compressor,dataset,uncompressed_bit_size,num_elements,compressed_bit_size,compression_ratio,residuals_bit_size,coefficients_bit_size,model_types_bit_size,starting_positions_bit_size,bpc_bit_size" << std::endl;
    std::cout << "NeaTS," << fn << "," << uncompressed_bit_size << "," << data.size() << ","
              << compressed_bit_size << ","
              << (long double) (compressed_bit_size) / (long double)(uncompressed_bit_size) << ",";

    std::visit([&](auto &&neats_compressor) {
        neats_compressor.size_info();
    }, compressor);

    std::visit([&](auto &&neats_compressor) {
        neats_compressor.to_csv(neats_fn);
    }, compressor);

    auto la_map = init_lavector<x_t, y_t, poly, T1, T2>(std::make_integer_sequence<int64_t, 40>{});
    auto la_compressor = la_map[bpc];

    std::visit([&](auto &&la_compressor) {
        la_compressor.partitioning(data.begin(), data.end());
    }, la_compressor);

    typename std::decay<decltype(data)>::type decompressed_lavector(data.size());
    std::visit([&](auto &&la_compressor) {
        la_compressor.decompress(std::forward<typename std::decay<decltype(data)>::type::iterator>(decompressed_lavector.begin()),
                                 std::forward<typename std::decay<decltype(data)>::type::iterator>(decompressed_lavector.end()));
    }, la_compressor);

    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed_lavector[i]) {
            std::cout << i << ": " << data[i] << "!=" << decompressed_lavector[i] << std::endl;
            //exit(1);
        }
    }

    auto compressed_bit_size_lavector = std::visit([](auto &&la_compressor) {
        return la_compressor.size_in_bits();
    }, la_compressor);

    std::cout << "LA_vector," << fn << "," << uncompressed_bit_size << "," << data.size() << ","
              << compressed_bit_size_lavector << ","
              << (double) (compressed_bit_size_lavector) / (double)(uncompressed_bit_size) << ",";

    std::visit([&](auto &&la_compressor) {
        la_compressor.size_info();
    }, la_compressor);

    std::visit([&](auto &&la_compressor) {
        la_compressor.to_csv(la_fn);
    }, la_compressor);

}


int main(int argc, char *argv[]) {

    int opt;
    std::string filename = "";
    bool float_type = true;
    bool long_poly = false;
    uint8_t bpc = 0;

    option opts[] = {
            { "help", no_argument, NULL, 0},
            { "file", required_argument, NULL, 'f' },
            { "T1", optional_argument, NULL, 'd' },
            { "error_bound", optional_argument, NULL, 'e' },
            { "long_poly", optional_argument, NULL, 'l' }
    };

    while ((opt = getopt_long(argc, argv, "hf:d::e::l::", opts, NULL))!=-1) {

        switch (opt) {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [-h] [-f filename] [-d coefficient_type] [-e error_bound]" << std::endl;
                return 0;
            case 'e':
                bpc = (uint8_t) std::stoi(optarg);
                break;
            case 'f':
                filename = std::string(optarg);
                //strcpy(filename, optarg, sizeof(filename));
                //filename[sizeof(filename) - 1] = '\0';
                break;
            case 'd':
                float_type = false;
                break;
            case 'l':
                long_poly = true;
                break;
            case '?':
                std::cerr << "Unknown option: " << optopt << std::endl;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-h] [-f filename] [-d coefficient_type] [-e error_bound]" << std::endl;
                break;
        }
    }

    if (filename.empty()) {
        std::cerr << "Usage: " << argv[0] << " [-h] [-f filename] [-t coefficient_type] [-e error_bound]" << std::endl;
        return 1;
    }

    if (float_type) {
        if (long_poly)
            run<uint32_t, int64_t, long double, float, double>(filename, bpc);
        else
            run<uint32_t, int64_t, double, float, double>(filename, bpc);
    } else {
        if (long_poly)
            run<uint32_t, int64_t, long double, double, double>(filename, bpc);
        else
            run<uint32_t, int64_t, double, double, double>(filename, bpc);
    }

    return 0;
}