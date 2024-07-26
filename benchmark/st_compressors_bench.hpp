#pragma once
#include "lib/BitStream.hpp"
#include "lib/Window.hpp"
#include "lib/Zigzag.hpp"
#include "Chimp/CompressorChimp.hpp"
#include "Chimp/DecompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "Chimp128/DecompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "TSXor/DecompressorTSXor.hpp"
#include "Gorilla/CompressorGorilla.hpp"
#include "Gorilla/DecompressorGorilla.hpp"
#include "utils.hpp"
#include <chrono>


template<typename T = double>
void gorilla_compression(const std::string &in_filename, std::ostream &out, size_t block_size = 1000) {
    const auto data = fa::utils::read_data_binary<T, T>(in_filename, !std::is_floating_point_v<T>);
    const auto n = data.size();

    const auto num_blocks = (n / block_size) + (n % block_size != 0);

    size_t total_compressed_size = 0;
    double compression_time = 0;
    double decompression_time = 0;

    for (auto ib = 0; ib < num_blocks; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);
        const auto data_block = std::vector<T>(data.begin() + (ib * (int64_t) block_size),
                                               data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto gorilla_compressor = CompressorGorilla<T>(*data_block.begin());
        for (auto it = (data_block.begin() + 1); it < data_block.end(); ++it) {
            gorilla_compressor.addValue(*it);
        }
        gorilla_compressor.close();
        auto t2 = std::chrono::high_resolution_clock::now();
        compression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        total_compressed_size += gorilla_compressor.bs_values.size(); // size in bits

        t1 = std::chrono::high_resolution_clock::now();
        auto gorilla_decompressor = DecompressorGorilla<T>(gorilla_compressor.bs_values, data_block.size());
        std::vector<T> decompressed_data(data_block.size());
        auto i = 0;
        decompressed_data[i] = gorilla_decompressor.storedValue;
        while (gorilla_decompressor.hasNext()) {
            decompressed_data[++i] = gorilla_decompressor.storedValue;
        }
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed_data);
        decompression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    out << n << ','; // #values
    out << static_cast<long double>(total_compressed_size) / n << ','; // #bits_per_value
    out << ((double) n * sizeof(T) / 1e+6) / (compression_time / 1e+9) << ','; // compression_speed
    out << ((double) n * sizeof(T) / 1e+6) / (decompression_time / 1e+9) << std::endl; // decompression_speed
}

void tsxor_compression(const std::string &filename, std::ostream &out, size_t block_size = 1000) {
    const auto data = fa::utils::read_data_binary<double, double>(filename, false);

    const auto n = data.size();

    const auto num_blocks = (n / block_size) + (n % block_size != 0);

    size_t total_compressed_size = 0;
    double compression_time = 0;
    double decompression_time = 0;

    for (auto ib = 0; ib < num_blocks; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);

        const auto data_block = std::vector<double>(data.begin() + (ib * (int64_t) block_size),
                                                    data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto tsxor_compressor = CompressorTSXor<double>(*data_block.begin());
        for (auto i = (data_block.begin() + 1); i < data_block.end(); ++i) {
            tsxor_compressor.addValue(*i);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        compression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        total_compressed_size += tsxor_compressor.bytes.size() * 8; // size in bits

        t1 = std::chrono::high_resolution_clock::now();
        auto tsxor_decompressor = DecompressorTSXor<double>(tsxor_compressor.bytes, data_block.size());
        auto i = 0;
        std::vector<double> decompressed_data(data_block.size());
        decompressed_data[i] = tsxor_decompressor.storedValue;
        while (tsxor_decompressor.hasNext()) {
            decompressed_data[++i] = tsxor_decompressor.storedValue;
        }
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed_data);
        decompression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

    }

    out << n << ','; // #values
    out << static_cast<double>(total_compressed_size) / n << ','; // #bits_per_value
    out << ((double) n * sizeof(double) / 1e+6) / (compression_time / 1e+9) << ','; // compression_speed
    out << ((double) n * sizeof(double) / 1e+6) / (decompression_time / 1e+9) << std::endl; // decompression_speed
}

void chimp_compression(const std::string &filename, std::ostream &out, size_t block_size = 1000) {
    const auto data = fa::utils::read_data_binary<double, double>(filename, false);

    const auto n = data.size();

    const auto num_blocks = (n / block_size) + (n % block_size != 0);

    size_t total_compressed_size = 0;
    double compression_time = 0;
    double decompression_time = 0;

    for (auto ib = 0; ib < num_blocks; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);

        const auto data_block = std::vector<double>(data.begin() + (ib * (int64_t) block_size),
                                                    data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto chimp_compressor = CompressorChimp<double>(*data_block.begin());
        for (auto i = (data_block.begin() + 1); i < data_block.end(); ++i) {
            chimp_compressor.addValue(*i);
        }
        chimp_compressor.close();
        auto t2 = std::chrono::high_resolution_clock::now();
        compression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        total_compressed_size += chimp_compressor.getSize(); // size in bits

        t1 = std::chrono::high_resolution_clock::now();
        auto chimp_decompressor = DecompressorChimp<double>(chimp_compressor.getBuffer(), data_block.size());
        auto i = 0;
        std::vector<double> decompressed_data(data_block.size());
        decompressed_data[i] = chimp_decompressor.storedValue;
        while (chimp_decompressor.hasNext()) {
            decompressed_data[++i] = chimp_decompressor.storedValue;
        }
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed_data);

        decompression_time += duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    out << n << ','; // #values
    out << static_cast<double>(total_compressed_size) / n << ','; // bits_per_value
    out << ((double) n * sizeof(double) / 1e+6) / (compression_time / 1e+9) << ','; // compression_speed
    out << ((double) n * sizeof(double) / 1e+6) / (decompression_time / 1e+9) << std::endl; // decompression_speed
}

void chimp128_compression(const std::string &filename, std::ostream &out, size_t block_size = 1000) {

    const auto data = fa::utils::read_data_binary<double, double>(filename, false);
    const auto n = data.size();
    const auto num_blocks = (n / block_size) + (n % block_size != 0);

    size_t total_compressed_size = 0;
    double compression_time = 0;
    double decompression_time = 0;

    for (auto ib = 0; ib < num_blocks; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);

        const auto data_block = std::vector<double>(data.begin() + (ib * (int64_t) block_size),
                                                    data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto chimp128_compressor = CompressorChimp128<double>(*data_block.begin());
        for (auto i = (data_block.begin() + 1); i < data_block.end(); ++i) {
            chimp128_compressor.addValue(*i);
        }
        chimp128_compressor.close();

        auto t2 = std::chrono::high_resolution_clock::now();
        compression_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        total_compressed_size += chimp128_compressor.getSize(); // size in bits

        auto chimp128_decompressor = DecompressorChimp128<double>(chimp128_compressor.getBuffer(),
                                                                  data_block.size());
        t1 = std::chrono::high_resolution_clock::now();
        auto i = 0;
        std::vector<double> decompressed_data(data_block.size());
        decompressed_data[i] = chimp128_decompressor.storedValue;
        while (chimp128_decompressor.hasNext()) {
            decompressed_data[++i] = chimp128_decompressor.storedValue;
        }
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed_data);
        decompression_time += duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        /*
        for (auto j = 0; j < data_block.size(); ++j) {
            if (data[j] != decompressed_data[j]) {
                std::cout << "i: " << j << ", " << decompressed_data[j] << ", " << data_block[j] << std::endl;
                //throw std::runtime_error("Error during decompression!");
            }
        }
        */

    }

    out << n << ','; // #values
    out << static_cast<double>(total_compressed_size) / n << ','; // #bits_per_value
    out << ((double) n * sizeof(double) / 1e+6) / (compression_time / 1e+9) << ','; // compression_speed
    out << ((double) n * sizeof(double) / 1e+6) / (decompression_time / 1e+9) << std::endl; // decompression_speed
}

template<typename T = CompressorChimp<double>, typename U = DecompressorChimp<double>>
void streaming_compressors_random_access(const std::string &in_fn,
                                         std::ostream &out, size_t block_size = 1000) {

    const auto data = fa::utils::read_data_binary<double, double>(in_fn, false);
    const auto n = data.size(); // number of values
    const auto num_blocks = (n / block_size) + (n % block_size != 0);

    std::vector<std::unique_ptr<T>> compressed_data_blocks{};
    size_t total_compressed_size = 0;
    for (auto ib = 0; ib < num_blocks; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<double>(data.begin() + (ib * (int64_t) block_size),
                                              data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        auto cmpr = T(*data_block.begin());
        for (auto it = (data_block.begin() + 1); it < data_block.end(); ++it) {
            cmpr.addValue(*it);
        }
        cmpr.close();
        total_compressed_size += cmpr.getSize();
        compressed_data_blocks.emplace_back(std::make_unique<T>(cmpr));
    }

    // Generating datasets of integers for the queries
    size_t num_queries = 1e+6;
    std::mt19937 mt1(2323);

    // select query
    std::uniform_int_distribution<size_t> dist1(1, n - 1);
    std::vector<size_t> indexes(num_queries);
    for (auto i = 0; i < num_queries; ++i) {
        indexes[i] = (dist1(mt1));
    }

    auto dcmpr = U(compressed_data_blocks[0]->getBuffer(), 1000);

    long time = 0;
    for (unsigned long index: indexes) {
        auto ib = index / block_size;
        auto offset = index % block_size;
        auto bs = std::min(block_size, n - ib * block_size);
        auto cmpr = *compressed_data_blocks[ib];
        auto t1 = std::chrono::high_resolution_clock::now();
        auto decmpr = U(cmpr.getBuffer(), bs);
        auto i = 0;
        while (i < offset && decmpr.hasNext()) {
            ++i;
        }
        auto value = decmpr.storedValue;
        auto t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(value);
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        if (data[index] != value) {
            std::cerr << data[index] << ", " << value << std::endl;
        }
    }

    double dt = (static_cast<double>(num_queries) * sizeof(int64_t) / 1e+6) / (static_cast<double>(time) / 1e+9);
    auto bpv = static_cast<double>(total_compressed_size) / n;
    out << bpv << "," << dt << std::endl;
}
