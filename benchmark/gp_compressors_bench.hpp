#pragma once

#include "utils.hpp"
#include <squash-0.7/squash/squash.h>

template<class T>
void do_not_optimize(T const &value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<class T>
const auto to_bytes = [](auto &&x) -> std::array<uint8_t, sizeof(T)> {
    std::array<uint8_t, sizeof(T)> arrayOfByte{};
    for (auto i = 0; i < sizeof(T); i++)
        arrayOfByte[(sizeof(T) - 1) - i] = (x >> (i * sizeof(T)));
    return arrayOfByte;
};


void squash_random_access(const std::string &compressor, const std::string &filename,
                          std::ostream &out, size_t block_size = 1000, int level = -1) {

    using T = int64_t;

    SquashCodec *codec = squash_get_codec(compressor.c_str());
    if (codec == nullptr) {
        throw std::runtime_error("Unable to find algorithm" + std::string(compressor));
    }

    SquashOptions *opts = nullptr;
    if (level != -1) {
        char level_s[4];
        opts = squash_options_new(codec, NULL);
        squash_object_ref_sink(opts);
        snprintf (level_s, 4, "%d", level);
        auto res_parse = squash_options_parse_option(opts, "level", level_s);

        if (res_parse != SQUASH_OK) {
            throw std::runtime_error("Unable to set level: " + std::to_string(level));
            exit(-1);
        }
    }

    const auto data = fa::utils::read_data_binary<T, T>(filename);
    const size_t n = data.size(); // number of values
    const auto num_blocks = n / block_size;

    size_t total_compressed_size = 0;
    std::vector<std::pair<uint8_t *, size_t>> compressed_data_blocks(num_blocks + 1);
    for (auto ib = 0; ib < num_blocks + 1; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<int64_t>(data.begin() + (ib * (int64_t) block_size),
                                               data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        std::vector<uint8_t> data_block_bytes;
        std::for_each(data_block.begin(), data_block.end(), [&](const auto &y) {
            auto bts = to_bytes<T>(y);
            data_block_bytes.insert(data_block_bytes.end(), bts.begin(), bts.end());
        });

        size_t uncompressed_size_in_bytes = data_block_bytes.size();
        size_t compressed_size_in_bytes = squash_get_max_compressed_size(compressor.data(),
                                                                         uncompressed_size_in_bytes);
        auto *compressed_data = (uint8_t *) malloc(compressed_size_in_bytes);

        SquashStatus res = squash_compress(compressor.data(), &compressed_size_in_bytes, compressed_data,
                                           uncompressed_size_in_bytes, data_block_bytes.data(), nullptr);

        compressed_data_blocks[ib] = {compressed_data, compressed_size_in_bytes};

        if (res != SQUASH_OK) {
            throw std::runtime_error("Unable to compress data: " + filename + " with compressor: " + compressor);
        }
        total_compressed_size += compressed_size_in_bytes * CHAR_BIT;
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

    size_t time = 0;
    for (unsigned long index: indexes) {
        auto ib = index / block_size;
        auto offset = index % block_size;
        auto bs = std::min(block_size, n - ib * block_size);
        size_t decompressed_size_in_bytes = bs * sizeof(int64_t) + 1;
        size_t compressed_size_in_bytes = compressed_data_blocks[ib].second;

        auto *decompressed_data = (uint8_t *) malloc(decompressed_size_in_bytes);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto res = squash_decompress(compressor.data(), &decompressed_size_in_bytes, (uint8_t *) decompressed_data,
                                     compressed_size_in_bytes,
                                     (uint8_t *) compressed_data_blocks[ib].first,
                                     nullptr);

        auto t2 = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        decompressed_data[decompressed_size_in_bytes] = '\0';
        do_not_optimize(decompressed_data);
        if (res != SQUASH_OK) {
            throw std::runtime_error("Unable to decompress data: " + filename + " with compressor: " + compressor);
        }

        int64_t value = 0;
        for (auto j = 0; j < sizeof(int64_t); j++) {
            value = (value << 8) + decompressed_data[(offset * sizeof(int64_t)) + j];
        }

        if (data[index] != value) throw std::runtime_error("Error during random access");
        free(decompressed_data);
    }

    for (auto &compressed_data_block: compressed_data_blocks) {
        free(compressed_data_block.first);
    }

    if (opts != nullptr) {
        squash_object_unref(opts);
    }

    out << time / num_queries << std::endl;
}


void squash_full(const std::string &compressor,
                 const std::string &filename,
                 std::ostream &out,
                 size_t block_size = 1000,
                 int level = -1) {

    using T = int64_t;

    SquashCodec *codec = squash_get_codec(compressor.c_str());
    if (codec == nullptr) {
        throw std::runtime_error("Unable to find algorithm" + std::string(compressor));
    }

    SquashOptions *opts = nullptr;
    if (level != -1) {
        char level_s[4];
        opts = squash_options_new(codec, NULL);
        squash_object_ref_sink(opts);
        snprintf (level_s, 4, "%d", level);
        auto res_parse = squash_options_parse_option(opts, "level", level_s);

        if (res_parse != SQUASH_OK) {
            throw std::runtime_error("Unable to set level: " + std::to_string(level));
            exit(-1);
        }
    }

    auto data = fa::utils::read_data_binary<T, T>(filename);
    const size_t n = data.size(); // number of values
    const auto num_blocks = n / block_size;

    size_t total_compressed_size = 0;
    size_t compression_time_ns = 0;
    size_t decompression_time_ns = 0;

    for (auto ib = 0; ib < num_blocks + 1; ++ib) {
        const auto bs = std::min(block_size, n - ib * block_size);
        auto data_block = std::vector<T>(data.begin() + (ib * (int64_t) block_size),
                                         data.begin() + (ib * (int64_t) block_size) + (int64_t) bs);

        std::vector<uint8_t> data_block_bytes;
        std::for_each(data_block.begin(), data_block.end(), [&](const auto &y) {
            auto bts = to_bytes<T>(y);
            data_block_bytes.insert(data_block_bytes.end(), bts.begin(), bts.end());
        });

        auto t1 = std::chrono::high_resolution_clock::now();
        size_t uncompressed_size_in_bytes = data_block_bytes.size();
        size_t compressed_size_in_bytes = squash_get_max_compressed_size(compressor.data(),
                                                                         uncompressed_size_in_bytes);
        auto *compressed_data = (uint8_t *) malloc(compressed_size_in_bytes);

        SquashStatus res = squash_compress_with_options(compressor.data(), &compressed_size_in_bytes, compressed_data,
                                           uncompressed_size_in_bytes, data_block_bytes.data(), opts);
        auto t2 = std::chrono::high_resolution_clock::now();
        compression_time_ns += duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        if (res != SQUASH_OK) {
            throw std::runtime_error("Unable to compress data: " + filename + " with compressor: " + compressor);
        }

        total_compressed_size += compressed_size_in_bytes * CHAR_BIT;

        t1 = std::chrono::high_resolution_clock::now();
        size_t decompressed_size_in_bytes = uncompressed_size_in_bytes + 1;
        auto *decompressed_data = (uint8_t *) malloc(decompressed_size_in_bytes);
        res = squash_decompress(compressor.data(), &decompressed_size_in_bytes, (uint8_t *) decompressed_data,
                                compressed_size_in_bytes,
                                compressed_data, opts);
        decompressed_data[decompressed_size_in_bytes] = '\0';
        t2 = std::chrono::high_resolution_clock::now();
        do_not_optimize(decompressed_data);
        if (res != SQUASH_OK) {
            throw std::runtime_error("Unable to decompress data: " + filename);
        }

        decompression_time_ns += duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        for (auto i = 0; i < data_block_bytes.size(); i += sizeof(T)) {
            //convert char array to T
            T value = 0;
            for (auto j = 0; j < sizeof(T); j++) {
                value = (value << 8) + decompressed_data[i + j];
            }

            if (value != data_block[i / sizeof(T)]) {
                std::cerr << "i: " << i << ", " << value << ", " << data_block[i] << std::endl;
                throw std::runtime_error("Error during decompression!");
            }
        }

        free(compressed_data);
        free(decompressed_data);
    }

    if (opts != nullptr) {
        squash_object_unref(opts);
    }
    // out << n << ','; // #values
    out << compressor << ','; // compressor
    out << level << ','; // level
    out << filename << ','; // dataset
    out << block_size << ','; // block_size
    out << n * sizeof(T) * 8 << ','; // uncompressed_bit_size
    out << total_compressed_size << ','; // compressed_storage_bit_size
    out << compression_time_ns << ','; // compression_time(ns)
    out << decompression_time_ns << ','; // decompression_time(ns)
    out << static_cast<long double>(total_compressed_size) / static_cast<long double>(n * sizeof(T) * 8) ; // compression_ratio
}
