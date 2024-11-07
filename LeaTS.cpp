#include <iostream>
#include <cmath>
#include <stdfloat>
#include "include/LeaTS.hpp"
#include <experimental/simd>

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

auto random_access_time(const auto &compressor, uint32_t num_runs = 10) {
    auto seed_val = 2323;
    std::mt19937 mt1(seed_val);
    size_t dt = 0;
    for (auto j = 0; j < num_runs; ++j) {
        size_t num_queries = 1e+6;
        // select query
        std::uniform_int_distribution<size_t> dist1(1, compressor.size() - 1);
        std::vector<size_t> indexes(num_queries);
        for (auto i = 0; i < num_queries; ++i) {
            indexes[i] = (dist1(mt1));
        }

        auto cnt = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (auto it = indexes.begin(); it < indexes.end(); ++it) {
            cnt += compressor[*it];
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(cnt);
        auto time = duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        dt += (time / num_queries);
    }
    return dt / num_runs;
};

template <typename T>
struct AlignedAllocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        return static_cast<T*>(std::aligned_alloc(sizeof(T) * 8, sizeof(T) * n));
    }
    void deallocate(T* p, std::size_t n) {
        std::free(p);
    }
};

template<typename T = int64_t >
double full_decompression_time(auto &compressor, uint32_t num_runs = 50) {
    //std::cout << "compressor size: " << compressor.size() << std::endl;
    size_t res{0};
    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < num_runs; ++i) {
        std::vector<T, AlignedAllocator<T>> decompressed(compressor.size());
        //compressor.decompress(decompressed.begin(), decompressed.end());
        compressor.simd_decompress(decompressed.data());
        //compressor.simd_unpack_
        do_not_optimize(decompressed);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ns_int = duration_cast<std::chrono::nanoseconds>(t2 - t1);
    res += ns_int.count();
    return ((double) res) / num_runs;
};

template<typename TypeIn = int64_t, typename TypeOut = int64_t, typename poly_t = double, typename T1 = std::float32_t, typename T2 = std::float64_t, typename x_t = uint32_t>
void inline run(const std::string &full_fn, int64_t bpc = 0, bool first_is_size = true, bool header = false) {
    //auto processed_data = pfa::algorithm::io::preprocess_data<TypeIn, TypeOut>(full_fn, 0, first_is_size);
    auto data = pfa::algorithm::io::read_data_binary<TypeIn, TypeOut>(full_fn, first_is_size);
    pfa::leats::compressor<x_t, TypeOut, poly_t, T1, T2> compressor{(uint8_t) bpc};
    auto t1 = std::chrono::high_resolution_clock::now();
    compressor.partitioning(data.begin(), data.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    auto compressed_size = compressor.size_in_bits();
    auto uncompressed_size = data.size() * sizeof(TypeIn) * 8;
    auto compression_ratio = (double) compressed_size / (double) uncompressed_size;
    auto compression_speed = (double) ((uncompressed_size / 8) / 1e6) / (compression_time / 1e9);

    //compressor.size_info(header);

    //auto decompressed = decltype(data)(compressor.size());
    //compressor.decompress(decompressed.begin(), decompressed.end());

    std::vector<TypeOut, AlignedAllocator<TypeOut>> decompressed(compressor.size());
    //compressor.decompress(decompressed.begin(), decompressed.end());
    compressor.simd_decompress(decompressed.data());

    auto num_errors = 0;
    int64_t max_error = 0;
    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            num_errors++;
            max_error = std::max(max_error, std::abs(data[i] - decompressed[i]));
        }
    }

    if (num_errors > 0) {
        std::cout << "Number of errors: " << num_errors << ", _MAX error: " << max_error << std::endl;
    }

    for (auto i = 0; i < data.size(); ++i) {
        if (std::abs(data[i] - decompressed[i]) > 1) {
            std::cout << "Error during RA at position " << i << ": " << data[i] << " != " << decompressed[i] << std::endl;
            break;
        }
    }

    auto random_access_t = random_access_time(compressor);
    auto random_access_speed = (double) (8 / 1e6 ) / (random_access_t / 1e9);

    auto full_decompression_t = full_decompression_time<TypeOut>(compressor);
    auto full_decompression_speed = ((uncompressed_size / 8) / 1e6) / (full_decompression_t / 1e9);
    //if (header) {
        std::cout << "compressor,dataset,compressed_bit_size,compression ratio,compression_speed(MB/s),random_access_speed(MB/s),full_decompression_speed(MB/s)" << std::endl;
        std::cout << "LeaTS," << full_fn << "," << compressed_size << "," << compression_ratio << ",";
        std::cout << compression_speed << ",";
        std::cout << random_access_speed << ", ";
        std::cout << full_decompression_speed << std::endl;
    //}
}

void println(std::string_view name, auto const& a)
{
    std::cout << name << ": ";
    for (std::size_t i{}; i != std::size(a); ++i)
        std::cout << a[i] << ' ';
    std::cout << '\n';
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <full_fn> [bpc] [first_is_size]" << std::endl;
        return 1;
    }

    // NeaTS (average metrics) vs LeaTS (average metrics)
    auto full_fn = std::string(argv[1]);
    auto bpc = std::stoi(argv[2]);

    //run<int64_t, int64_t, double, std::float32_t, std::float64_t, uint32_t>(fn, bpc, true);


    /*
    namespace stdx = std::experimental;
    std::vector<double> a{3.2, 4.1, -2.2, 0.0, 1.1, 7.87, 33.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};

    const stdx::native_simd<double> b([&a](double i) { return a[i]; });
    println("b", b);
    */

    std::string path = "/data/citypost/neat_datasets/binary/big/";
    run<int64_t, int64_t, double, float, float>(std::string(full_fn), uint8_t(bpc), true);

    //run<int64_t, int64_t, double, std::float32_t, std::float32_t>(path + std::string("wind-dir.bin"), 16);
    /*
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("dew-point-temp.bin"), 12);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("air-pressure.bin"), 15);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("wind-dir.bin"), 16);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("germany.bin"), 13);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("uk.bin"), 12);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("usa.bin"), 10);
    run<int64_t, int64_t, double, std::float32_t, std::float64_t>(path + std::string("IR-temp.bin"), 12);

    run<int64_t, int64_t, double, std::float64_t, std::float64_t>(path + std::string("geolife-lat.bin"), 20);
    run<int64_t, int64_t, double, std::float64_t, std::float64_t>(path + std::string("geolife-lon.bin"), 20);
    */

    //run<10, double, float, double>(path + std::string("city_temperature.bin"));
    //run<12, double, float, double>(path + std::string("dew-point-temp.bin"));
    /*
    run<15, double, float, double>(path + std::string("air-pressure.bin"));
    run<16, double, float, double>(path + std::string("wind-dir.bin"));
    run<13, double, float, double>(path + std::string("germany.bin"));
    run<12, double, float, double>(path + std::string("uk.bin"));
    run<10, double, float, double>(path + std::string("usa.bin"));
    run<12, double, float, double>(path + std::string("IR-temp.bin"));

    run<20, long double, double, double>(path + std::string("geolife-lat.bin"));
    run<20, long double, double, double>(path + std::string("geolife-lon.bin"));

    run<22, double, float, double>(path + std::string("bird-migration.bin"));
    run<24, double, float, double>(path + std::string("bitcoin-price.bin"));
    run<37, long double, double, double>(path + std::string("basel-temp.bin"));
    run<30, long double, double, double>(path + std::string("basel-wind.bin"));
    */
    return 0;
}
