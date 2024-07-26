#include <iostream>
#include <cmath>
#include <stdfloat>
#include "include/NeaTS.hpp"
#include <sdsl/construct.hpp>
#include <filesystem>
#include <experimental/simd>
#include <fstream>

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
        return static_cast<T*>(std::aligned_alloc(64, sizeof(T) * n));
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
    auto cnt = 0;
    for (auto i = 0; i < num_runs; ++i) {
        std::vector<T, AlignedAllocator<T>> decompressed(compressor.size());
        compressor.simd_decompress(decompressed.data());
        do_not_optimize(decompressed);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    do_not_optimize(cnt);
    auto ns_int = duration_cast<std::chrono::nanoseconds>(t2 - t1);
    res += ns_int.count();
    return ((double) res) / num_runs;
};

template<typename TypeIn = int64_t, typename TypeOut = int64_t, typename poly_t = double, typename T1 = std::float32_t, typename T2 = std::float64_t, typename x_t = uint32_t>
void inline run(const std::string &full_fn, int64_t bpc = 0, bool first_is_size = true, bool header = false) {
    //auto processed_data = pfa::algorithm::io::preprocess_data<TypeIn, TypeOut>(full_fn, 0, first_is_size);
    auto data = pfa::algorithm::io::preprocess_data<TypeIn>(full_fn, bpc, first_is_size);

    pfa::neats::compressor<x_t, TypeOut, poly_t, T1, T2> compressor{(uint8_t) bpc};
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

    std::vector<int64_t, AlignedAllocator<int64_t>> decompressed(compressor.size());
    //compressor.decompress(decompressed.begin(), decompressed.end());
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
        std::cout << "Number of errors during decompression: " << num_errors << ", _MAX error: " << max_error << std::endl;
    }

    num_errors = 0;
    max_error = 0;
    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != compressor[i]) {
            std::cout << "Error during Random Access at index: " << i << ", expected: " << data[i] << ", got: " << compressor[i] << std::endl;
            num_errors++;
            max_error = std::max(max_error, std::abs(data[i] - compressor[i]));
        }
    }

    if (num_errors > 0) {
        std::cout << "Number of errors during RA: " << num_errors << ", _MAX error: " << max_error << std::endl;
    }

    auto random_access_t = random_access_time(compressor);
    auto random_access_speed = (double) (8 / 1e6 ) / (random_access_t / 1e9);

    auto full_decompression_t = full_decompression_time(compressor);
    auto full_decompression_speed = ((uncompressed_size / 8) / 1e6) / (full_decompression_t / 1e9);
    //if (header) {
        std::cout << "compressor,dataset,compressed_bit_size,compression ratio,compression_speed(MB/s),random_access_speed(MB/s),full_decompression_speed(MB/s)" << std::endl;
        std::cout << "NeaTS," << full_fn << "," << compressed_size << "," << compression_ratio << ",";
        std::cout << compression_speed << ",";
        std::cout << random_access_speed << ", ";
        std::cout << full_decompression_speed << std::endl;
    //}
}

void inline from_file(const std::string& original_fn, const std::string& neats_fn) {
    std::ifstream out(neats_fn, std::ios::binary | std::ios::in);
    auto compressor = pfa::neats::compressor<uint32_t, int64_t, double, std::float32_t, std::float32_t>::load(out);
    std::vector<int64_t> decompressed(compressor.size());
    compressor.decompress(decompressed.begin(), decompressed.end());
    std::cout << "decompressed size: " << decompressed.size() << std::endl;

    auto processed_data = pfa::algorithm::io::preprocess_data<int64_t, int64_t>(original_fn, compressor.bits_per_residual());
    for (auto i = 0; i < processed_data.size(); ++i) {
        if (processed_data[i] != compressor[i]) {
            std::cout << "Error at index: " << i << ", expected: " << processed_data[i] << ", got: " << decompressed[i] << std::endl;
            exit(-1);
        }
    }
}

/*
namespace stdx = std::experimental;
using floatv = stdx::native_simd<float>;
using doublev = stdx::rebind_simd_t<int64_t, floatv>;

// use double precision internally
doublev dp(floatv x) {
    return stdx::static_simd_cast<doublev>(stdx::ceil(x));
}
*/

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <full_fn> [bpc] [first_is_size]" << std::endl;
        return 1;
    }

    auto full_fn = std::string(argv[1]);
    auto bpc = std::stoi(argv[2]);
    //std::string path = "/data/citypost/neat_datasets/binary/big/";
    run<int64_t, int64_t, double, float, double>(std::string(full_fn), uint8_t(bpc), true);

    // double, double, double
    // basel-wind 30
    // basel-temp 37
    // bitcoin-price 24
    // bird-migration 22

    // long double, double, double
    // geolife-lat 20
    // geolife-lon 20

    // double, float, float
    // wind-dir 15
    // I.bin_int64 15

    // double, float, double
    // uk 9
    // germany 13


    /*
    namespace stdx = std::experimental;

    auto original_fn = std::string("/data/citypost/neat_datasets/binary/big/dew-point-temp.bin");
    auto neats_fn = std::string("/data/citypost/neat_datasets/binary/big/dew-point-temp.bin.neats");

    std::ifstream out(neats_fn, std::ios::binary | std::ios::in);
    auto compressor = pfa::neats::compressor<uint32_t, int64_t, double, float, float>::load(out);
    std::cout << "compressed size: " << compressor.size() << std::endl;
    //std::vector<int64_t> decompressed(compressor.size());
    //std::cout << "decompressed size: " << decompressed.size() << std::endl;

    auto processed_data = pfa::algorithm::io::preprocess_data<int64_t, int64_t>(original_fn, compressor.bits_per_residual());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::decay_t<decltype(t1)> t2, t3;
    t1 = std::chrono::high_resolution_clock::now();
    std::vector<double, AlignedAllocator<double>> residuals_decompressed(compressor.size());
    compressor.simd_unpack_residuals(residuals_decompressed.data());
        //compressor.simd_approximations(residuals_decompressed.data());
    //compressor.simd_decompress(residuals_decompressed.data());
    t2 = std::chrono::high_resolution_clock::now();

    //delete[] residuals_decompressed;

    //compressor.simd_approximations(residuals_decompressed.data());
        //compressor.simd_decompress(residuals_decompressed);
    //}
    t3 = std::chrono::high_resolution_clock::now();
    auto unpack_residuals_time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    auto unpack_approximations_time = (std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    auto unpack_time = unpack_residuals_time + unpack_approximations_time;
    t1 = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> decompressed(compressor.size(), 0);
    compressor.decompress(decompressed.begin(), decompressed.end());
    t2 = std::chrono::high_resolution_clock::now();

    auto decompression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    std::cout << "decompression time: " << decompression_time << " ns" << std::endl;
    std::cout << "unpack time: " << unpack_time << " ns" << std::endl;
    std::cout << "unpack residuals time: " << unpack_residuals_time << " ns" << std::endl;
    std::cout << "unpack approximations time: " << unpack_approximations_time << " ns" << std::endl;
    auto uncompressed_size_mb = (compressor.size() * sizeof(int64_t)) / 1e6;
    auto unpack_speed = (double) uncompressed_size_mb / (unpack_time / 1e9);
    auto decompression_speed = (double) uncompressed_size_mb / (decompression_time / 1e9);

    std::cout << "uncompressed size (MB): " << uncompressed_size_mb << std::endl;
    std::cout << "decompressed size (num elements): " << processed_data.size() << std::endl;
    std::cout << "decompression speed: " << decompression_speed << " MB/s" << std::endl;
    std::cout << "unpack speed: " << unpack_speed << " MB/s" << std::endl;
    auto num_errors = 0;

    //compressor.simd_approximations(residuals_decompressed);

    auto cnt = 0;
    for (auto i = 0; i < processed_data.size(); ++i) {

        if (processed_data[i] != compressor[i]) {
            std::cout << "Error during Random Access at index: " << i << ", expected: " << processed_data[i] << ", got: " << compressor[i] << std::endl;
            //exit(-1);
        }

        auto err = std::abs(processed_data[i] - (residuals_decompressed[i]));
        if (err > 1) {
            num_errors++;
            std::cout << "Error at index: " << i << ", expected: " << processed_data[i] << ", got: " << residuals_decompressed[i] << std::endl;
            std::cout << "Error: " << err << std::endl;
            if (cnt++ > 10) break;
        }
        //std::cout << "i: " << i << ", expected: " << processed_data[i] << ", got: " << decompressed[i] << " + " << residuals_decompressed[i] << std::endl;
        //if (i == 11) break;
    }

    std::cout << "Number of errors: " << num_errors << std::endl;
    //std::cout << *(residuals_decompressed.end() - 1) << std::endl;
    */

    // Decompression speed (LeaTS) [DONE] vs ALP [DONE]
    // Decompression speed (NeaTS) [DONE]
    // Make residuals simd NeaTS [DONE]
    // Take only one geolife dataset
    // Add one healtcare dataset
    // radar plot (cr, decompression speed, compression speed, random access speed) [LeaTS, NeaTS (e tutti gli altri)]
    // NeaTS sampling


    // LeaTS vs NeaTS
    // sampling vs compression ratio
    // commento real-time analysis da revisore 3 => non siamo "real-time"
    /*
    namespace stdx = std::experimental;

    using x_t = uint32_t;
    using xv_t = stdx::native_simd<x_t>;
    using y_t = int64_t;
    using yv_t = stdx::native_simd<y_t>;

    using T1 = std::float32_t;
    using T2 = std::float64_t;
    using max_t = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>;
    using floatv_t = stdx::native_simd<max_t>;
    constexpr std::size_t floatv_size = floatv_t::size();
    using uint64v_t = stdx::rebind_simd_t<uint64_t, xv_t>;

    std::string t0_str = std::string("23.0");
    std::string t1_str = std::string("17.7653");

    const floatv_t t0v{static_cast<max_t>(std::stod(t0_str))};
    const floatv_t t1v{static_cast<max_t>(std::stod(t1_str))};

    max_t start = 0;
    max_t x_v[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    max_t y_v[std::size(x_v)];

    floatv_t startv{start};
    for (std::size_t i{}; i < std::size(x_v); i += floatv_size) {
        floatv_t v(&x_v[i], stdx::element_aligned);
        v = (v - startv) * t0v + t1v;
        v.copy_to(&y_v[i], stdx::element_aligned);
    }

    println("x_v", x_v);
    println("y_v", y_v);
    */

    return 0;
}
