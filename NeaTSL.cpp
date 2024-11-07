#include <iostream>
#include "include/NeaTSL.hpp"
#include "benchmark/AdaptiveApproximation.hpp"

using x_t = uint32_t;
using y_t = int64_t;

template<int64_t error_bound, typename T0, typename T1, typename T2>
void lossy_compression(const auto &fn, std::ostream &out, bool first_size = true) {
    neats::lossy_compressor<x_t, y_t, error_bound, T0, T1, T2> lc;
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= (min_data - error_bound); });

    auto t1 = std::chrono::high_resolution_clock::now();
    lc.partitioning(data.begin(), data.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();


    //std::cout << "Number of segments: " << lc.num_partitions() << std::endl;
    std::size_t decompression_time;
    auto num_runs = 1000;
    t1 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < num_runs; ++i) {
       auto decompressed = lc.decompress();
    }
    t2 = std::chrono::high_resolution_clock::now();
    decompression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    decompression_time /= num_runs;

    auto mape = 0.0;

    auto decompressed = lc.decompress();
    auto zeros = 0;
    for (auto i = 0; i < data.size(); ++i) {
        mape += std::abs((data[i] - decompressed[i]) / (double) data[i]);
        zeros += std::abs((data[i] - decompressed[i])) == 0;

        auto err = data[i] - decompressed[i];
        if (err > error_bound || err < -(error_bound + 1)) {
            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
            exit(1);
        }
    }
    mape /= data.size();
    //std::cout << "Zeros: " << zeros << std::endl;
    //std::cout << "Models: " << lc.num_partitions() << std::endl;

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    auto compressed_bit_size = lc.size_in_bits();
    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);

    auto compression_speed = ((uncompressed_bit_size / 8) / 1e6) / (compression_time / 1e9);
    auto decompression_speed = ((uncompressed_bit_size / 8) / 1e6) / (decompression_time / 1e9);

    out << "NeaTS-Lossy," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << "," << compression_speed << "," << mape << ","
        << decompression_speed << std::endl;
}

template<int64_t error_bound, typename T0, typename T1, typename T2>
void pla_compression(const auto &fn, std::ostream &out, bool first_size = true) {
    using poa_t = pfa::piecewise_optimal_approximation<x_t, y_t, T0, T1, T2>;
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= min_data - error_bound; });
    auto t1 = std::chrono::high_resolution_clock::now();
    //auto res = pfa::algorithm::make_pla<typename poa_t::pla_t, error_bound>(data.begin(), data.end());
    auto pa = typename poa_t::pla_t{error_bound};
    auto res = pa.make_approximation(data.begin(), data.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    size_t decompression_time;
    auto num_runs = 1000;
    t1 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < num_runs; ++i) {
        auto _decompressed = pa.get_approximations(res, data.size());
    }
    t2 = std::chrono::high_resolution_clock::now();
    decompression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    decompression_time /= num_runs;

    auto decompressed = pa.get_approximations(res, data.size());
    auto mape = 0.0;
    for (auto i = 0; i < data.size(); ++i) {
        mape += std::abs((data[i] - decompressed[i]) / (double) data[i]);
        /*
        if (err > error_bound || err < -(error_bound + 1)) {
            std::cout << i << ": " << data[i] << "!=" << decompressed[i] << std::endl;
            exit(1);
        }
        */
    }
    mape /= data.size();

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    size_t compressed_bit_size = 0;
    compressed_bit_size += res.size() * sizeof(x_t) * 8; // starting positions // num pla
    compressed_bit_size += res.size() * sizeof(T1) * 8; // slope
    compressed_bit_size += res.size() * sizeof(T2) * 8; // intercept

    //std::cout << "Number of segments: " << res.size() << std::endl;

    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);
    auto compression_speed = ((uncompressed_bit_size / 8) / 1e6) / (compression_time / 1e9);
    auto decompression_speed = ((uncompressed_bit_size / 8) / 1e6) / (decompression_time / 1e9);

    out << "PLA," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << "," << compression_speed << "," << mape <<
        "," << decompression_speed << std::endl;
}


template<int64_t error_bound, typename T0, typename T1, typename T2>
void adaptive_approximation_full(const auto &fn, std::ostream &out, bool first_size = true) {
    auto data = pfa::algorithm::io::read_data_binary<y_t, y_t>(fn, first_size);
    auto range = *std::max_element(data.begin(), data.end()) - *std::min_element(data.begin(), data.end());
    auto min_data = *std::min_element(data.begin(), data.end());
    min_data = min_data < 0 ? (min_data - 1) : -1;
    std::for_each(data.begin(), data.end(), [min_data](auto &d) { d -= min_data - error_bound; });

    auto t1 = std::chrono::high_resolution_clock::now();
    auto AA = adaptive_approximation<T0, x_t, y_t, T1, T2>(data.begin(), data.end(), error_bound);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    auto num_runs = 1000;
    t1 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < num_runs; ++i) {
        auto decompressed = AA.decompress();
    }
    t2 = std::chrono::high_resolution_clock::now();
    auto decompression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    decompression_time /= num_runs;

    //AA.check_partitions(data.begin(), data.end());
    double mape = AA.mape(data.begin(), data.end());

    auto uncompressed_bit_size = data.size() * sizeof(y_t) * 8;
    auto compressed_bit_size = AA.size_in_bits();
    auto cr = (long double) (compressed_bit_size) / (long double) (uncompressed_bit_size);
    auto compression_speed = ((uncompressed_bit_size / 8) / 1e6) / (compression_time / 1e9);
    auto decompression_speed = ((uncompressed_bit_size / 8) / 1e6) / (decompression_time / 1e9);

    out << "AA," << fn << "," << error_bound << "," << uncompressed_bit_size << "," << data.size() << ","
        << compressed_bit_size << "," << cr << "," << range << "," << compression_speed << "," << mape << ","
        << decompression_speed << std::endl;
}

int main(int argc, char *argv[]) {
    auto path = std::string(argv[1]);
    //lossy_compression<27, double, float, double>(path + std::string("uk.bin"), std::cout);
    //lossy_compression<23, double, float, double>(path + std::string("IR-temp.bin"), std::cout);
    //adaptive_approximation_full<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout, true);

    //lossy_compression<27, double, float, double>(path + std::string("germany.bin"), std::cout, true);
    //pla_compression<27, double, float, double>(path + std::string("germany.bin"), std::cout, true);
    //adaptive_approximation_full<27, double, float, double>(path + std::string("germany.bin"), std::cout, true);
    //lossy_compression<2289, double, float, double>(path + std::string("wind-dir.bin"), std::cout);

    lossy_compression<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout);
    pla_compression<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout);
    /*
    lossy_compression<9, double, float, double>(path + std::string("dust.bin"), std::cout);
    pla_compression<9, double, float, double>(path + std::string("dust.bin"), std::cout);
    adaptive_approximation_full<9, double, float, double>(path + std::string("dust.bin"), std::cout);

    lossy_compression<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout);
    pla_compression<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout);
    adaptive_approximation_full<62, double, float, double>(path + std::string("city_temperature.bin"), std::cout);

    lossy_compression<68, double, float, double>(path + std::string("dew-point-temp.bin"), std::cout);
    pla_compression<68, double, float, double>(path + std::string("dew-point-temp.bin"), std::cout);
    adaptive_approximation_full<68, double, float, double>(path + std::string("dew-point-temp.bin"), std::cout);

    lossy_compression<27, double, float, double>(path + std::string("germany.bin"), std::cout);
    pla_compression<27, double, float, double>(path + std::string("germany.bin"), std::cout);
    adaptive_approximation_full<27, double, float, double>(path + std::string("germany.bin"), std::cout);

    lossy_compression<9, double, float, double>(path + std::string("uk.bin"), std::cout);
    pla_compression<9, double, float, double>(path + std::string("uk.bin"), std::cout);
    adaptive_approximation_full<9, double, float, double>(path + std::string("uk.bin"), std::cout);

    lossy_compression<5, double, float, double>(path + std::string("usa.bin"), std::cout);
    pla_compression<5, double, float, double>(path + std::string("usa.bin"), std::cout);
    adaptive_approximation_full<5, double, float, double>(path + std::string("usa.bin"), std::cout);

    lossy_compression<2289, double, float, double>(path + std::string("wind-dir.bin"), std::cout);
    pla_compression<2289, double, float, double>(path + std::string("wind-dir.bin"), std::cout);
    adaptive_approximation_full<2289, double, float, double>(path + std::string("wind-dir.bin"), std::cout);

    lossy_compression<334, double, float, double>(path + std::string("air-pressure.bin"), std::cout);
    pla_compression<334, double, float, double>(path + std::string("air-pressure.bin"), std::cout);
    adaptive_approximation_full<334, double, float, double>(path + std::string("air-pressure.bin"), std::cout);

    lossy_compression<23, double, float, double>(path + std::string("IR-temp.bin"), std::cout);
    pla_compression<23, double, float, double>(path + std::string("IR-temp.bin"), std::cout);
    adaptive_approximation_full<23, double, float, double>(path + std::string("IR-temp.bin"), std::cout);

    lossy_compression<285, long double, double, double>(path + std::string("geolife-lat.bin"), std::cout);
    pla_compression<285, long double, double, double>(path + std::string("geolife-lat.bin"), std::cout);
    adaptive_approximation_full<285, long double, double, double>(path + std::string("geolife-lat.bin"), std::cout);

    lossy_compression<50, long double, double, double>(path + std::string("geolife-lon.bin"), std::cout);
    pla_compression<50, long double, double, double>(path + std::string("geolife-lon.bin"), std::cout);
    adaptive_approximation_full<50, long double, double, double>(path + std::string("geolife-lon.bin"), std::cout);

    */

    /*
    lossy_compression<238820000, long double, double, double>(path + std::string("basel-temp.bin"), std::cout);
    pla_compression<238820000, long double, double, double>(path + std::string("basel-temp.bin"), std::cout);
    adaptive_approximation_full<238820000, long double, double, double>(path + std::string("basel-temp.bin"),
                                                                        std::cout);

    lossy_compression<12790000, long double, double, double>(path + std::string("basel-wind.bin"), std::cout);
    pla_compression<12790000, long double, double, double>(path + std::string("basel-wind.bin"), std::cout);
    adaptive_approximation_full<12790000, long double, double, double>(path + std::string("basel-wind.bin"), std::cout);
     */

    /*
    lossy_compression<266000, double, double, double>(path + std::string("bitcoin-price.bin"), std::cout);
    pla_compression<266000, double, double, double>(path + std::string("bitcoin-price.bin"), std::cout);
    adaptive_approximation_full<266000, double, double, double>(path + std::string("bitcoin-price.bin"), std::cout);

    lossy_compression<900, double, float, double>(path + std::string("bird-migration.bin"), std::cout);
    pla_compression<900, double, float, double>(path + std::string("bird-migration.bin"), std::cout);
    adaptive_approximation_full<900, double, float, double>(path + std::string("bird-migration.bin"), std::cout);
    */

    //lossy_compression<20, double, float, double>(path + std::string("I.bin_int64"), std::cout, false);
    //pla_compression<20, double, float, double>(path + std::string("I.bin_int64"), std::cout, false);
    //adaptive_approximation_full<20, double, float, double>(path + std::string("I.bin_int64"), std::cout, false);

}