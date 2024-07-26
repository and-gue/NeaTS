#include <iostream>
#include <cmath>
#include <stdfloat>
#include "include/SNeaTS.hpp"
#include "include/LeaTS.hpp"
#include <sdsl/construct.hpp>
#include <filesystem>
#include <experimental/simd>
#include <fstream>


#include <ranges>

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
    pfa::sneats::compressor<x_t, TypeOut, poly_t, T1, T2> compressor{(uint8_t) bpc};
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
        std::cout << "Number of errors: " << num_errors << ", _MAX error: " << max_error << std::endl;
    }

    for (auto i = 0; i < data.size(); ++i) {
        if (data[i] != decompressed[i]) {
            std::cout << "Error during decompression at position " << i << ": " << data[i] << " != " << decompressed[i] << std::endl;
            break;
        }
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
    auto compressor = pfa::sneats::compressor<uint32_t, int64_t, double, std::float32_t, std::float32_t>::load(out);
    std::vector<int64_t> decompressed(compressor.size());
    compressor.simd_decompress(decompressed.data());
    std::cout << "decompressed size: " << decompressed.size() << std::endl;

    auto processed_data = pfa::algorithm::io::preprocess_data<int64_t, int64_t>(original_fn, compressor.bits_per_residual());
    for (auto i = 0; i < processed_data.size(); ++i) {
        if (processed_data[i] != compressor[i]) {
            std::cout << "Error at index: " << i << ", expected: " << processed_data[i] << ", got: " << decompressed[i] << std::endl;
            exit(-1);
        }
    }
}


namespace stdx = std::experimental;

int main(int argc, char *argv[]) {

    // dataset, bpc, tsize, M
    // city_temperature.bin 14 8192 5
    // dew-point-temperature.bin 12 8192 5
    // germany.bin 13 8192 7

    // uk.bin 12 4096 6
    // usa.bin 10 65536 6?
    // wind-dir.bin 16 8192 5
    // air-pressure.bin 15 8192 5
    // IR-temp.bin 12 8192 5
    // geolife-lat.bin 20 8192 5
    // I.bin_int64 15 8192 5


    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <full_fn> [bpc] [first_is_size] [tsize]" << std::endl;
        return 1;
    }

    auto full_fn = std::string(argv[1]);
    auto bpc = std::stoi(argv[2]);
    auto tsize = std::stoi(argv[3]);

    using type_in = int64_t;
    using allocator_t = AlignedAllocator<type_in>;
    using simd_t = stdx::native_simd<type_in>;
    constexpr auto simd_width = simd_t::size();

    auto data_vec = pfa::algorithm::io::read_vector_binary<int64_t, int64_t>(full_fn, true);

    auto simd_min = [](auto&& ptr, uint32_t n){
       simd_t simd_w;
       simd_t::value_type min_val = 0;
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

    auto simd_preprocess = [](auto&& ptr, uint32_t n, type_in v) {
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

    auto epsilon = (type_in) (BPC_TO_EPSILON(bpc));

    auto t1 = std::chrono::high_resolution_clock::now();
    simd_preprocess(data_vec.data(), data_vec.size(), simd_min(data_vec.data(), data_vec.size()) - epsilon);
    auto t2 = std::chrono::high_resolution_clock::now();
    //auto preprocess_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    //std::cout << "Preprocess speed: " << ((data_vec.size() * sizeof(type_in)) / 1e6) / ((preprocess_time) / 1e9) << std::endl;
    std::cout << "compressor,dataset,compressed_bit_size,compression ratio,compression_speed(MB/s),training_size,M" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    auto [icmpr, ccmpr] = pfa::sneats::partitioning<uint32_t, type_in, double, double, double>(data_vec.begin(), data_vec.end(), bpc, tsize);
    t2 = std::chrono::high_resolution_clock::now();
    auto partitioning_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    auto compression_speed = ((data_vec.size() * sizeof(type_in)) / 1e6) / ((partitioning_time) / 1e9);
    //std::cout << "Partitioning speed: " << ((data_vec.size() * sizeof(type_in)) / 1e6) / ((partitioning_time) / 1e9) << std::endl;
    auto compressed_bit_size = icmpr.size_in_bits() + ccmpr.size_in_bits();
    auto compression_ratio = (double) compressed_bit_size / (double) (data_vec.size() * sizeof(type_in) * 8);
    //std::cout << "Compression ratio: " << compression_ratio << std::endl;
    std::cout << "NeaTS," << full_fn << "," << compressed_bit_size << "," << compression_ratio << "," << compression_speed << "," << tsize << "," << bpc << std::endl;

    std::vector<int64_t, AlignedAllocator<int64_t>> idecompressed(icmpr.size());
    icmpr.simd_decompress(idecompressed.data());

    for (auto i = 0; i < tsize; ++i) {
        if (data_vec[i] != idecompressed[i]) {
            std::cout << "Error at index: " << i << ", expected: " << data_vec[i] << ", got: " << idecompressed[i] << std::endl;
            //exit(-1);
        }
    }

    std::vector<int64_t, AlignedAllocator<int64_t>> cdecompressed(ccmpr.size());
    ccmpr.simd_decompress(cdecompressed.data());
    auto j = 0;
    for (auto i = tsize; i < data_vec.size(); ++i) {
        if (data_vec[i] != cdecompressed[j++]) {
            std::cout << "Error at index: " << i << ", expected: " << data_vec[i] << ", got: " << cdecompressed[i] << std::endl;
            exit(-1);
        }
    }

    /*
    auto start = data_vec.begin();
    auto end = data_vec.begin() + (131072 * 8);

    pfa::sneats::compressor<uint32_t, type_in, double, float, double> compressor{(uint8_t) bpc};
    t1 = std::chrono::high_resolution_clock::now();
    compressor.initial_partitioning(start, end);
    t2 = std::chrono::high_resolution_clock::now();
    auto compression_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    auto uncompressed_size = std::distance(start, end) * sizeof(type_in) * 8;
    auto compression_speed = (double) ((uncompressed_size / 8) / 1e6) / (compression_time / 1e9);

    start = end;
    end = data_vec.end();
    uncompressed_size += std::distance(start, end) * sizeof(type_in) * 8;

    pfa::sneats::compressor<uint32_t, type_in, double, float, double> t_compressor{(uint8_t) bpc};
    t1 = std::chrono::high_resolution_clock::now();
    t_compressor.custom_partitioning(start, end, compressor.best_models<5>());
    t2 = std::chrono::high_resolution_clock::now();
    auto custom_partitioning_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    auto compressed_size = t_compressor.size_in_bits() + compressor.size_in_bits();
    auto compression_ratio = (double) compressed_size / (double) uncompressed_size;
    auto custom_partitioning_speed =  (double) ((uncompressed_size / 8) / 1e6) / (custom_partitioning_time / 1e9);

    std::cout << "compressor,dataset,compressed_bit_size,compression ratio,compression_speed(MB/s),compression_speed_t" << std::endl;
    std::cout << "NeaTS," << full_fn << "," << compressed_size << "," << compression_ratio << "," << compression_speed << "," << custom_partitioning_speed << std::endl;

    std::vector<int64_t, AlignedAllocator<int64_t>> decompressed(t_compressor.size());
    t_compressor.simd_decompress(decompressed.data());

    auto num_errors = 0;
    int64_t max_error = 0;
    for (auto i = 0; i < std::distance(start, end); ++i) {
        if (data_vec[i + (131072 / 2)] != decompressed[i]) {
            num_errors++;
            max_error = std::max(max_error, std::abs(data_vec[i] - decompressed[i]));
        }
    }
    std::cout << "Number of errors: " << num_errors << ", _MAX error: " << max_error << std::endl;

    auto _models = compressor.best_models<10>();
    */

    //data_ptr =  simd_for(std::move(data_ptr), data.size(), print);


    // entro martedì
    // 1. test su healthcare data [DONE]
    // 2. sim-piece in tabella lossy
    // 3. radar plot
    //

    // Decompression speed (LeaTS) [DONE] vs ALP [DONE]
    // Decompression speed (NeaTS) [DONE]
    // Make residuals simd NeaTS [DONE]
    // Take only one geolife dataset
    // Add one healtcare dataset [DONE] (plot al posto di dew-point-temp (ECG-I))
    // radar plot (cr, decompression speed, compression speed, random access speed) [LeaTS, NeaTS (e tutti gli altri)]
    // NeaTS sampling
    // NeaTS e LeaTS sampling nel plot compression speed vs compression ratio
    // Sim-piece nella tabella lossy

    // LeaTS vs NeaTS
    // sampling vs compression ratio
    // commento real-time analysis da revisore 3 => non siamo "real-time"

    return 0;
}
