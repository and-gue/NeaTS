#pragma once

#include <limits>
#include <iostream>
#include "../lib/BitStream.hpp"

template<typename T>
class CompressorChimp {

    uint64_t storedLeadingZeros = std::numeric_limits<uint64_t>::max();
    T storedValue = 0;
    bool first = true;
    size_t size;
    constexpr static uint32_t THRESHOLD = 6;

    constexpr static uint16_t leadingRepresentation[] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7
    };

    constexpr static uint16_t leadingRound[] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            8, 8, 8, 8, 12, 12, 12, 12,
            16, 16, 18, 18, 20, 20, 22, 22,
            24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24
    };

    BitStream out{};

    inline void writeFirst(const T& value) {
        auto *x = (uint64_t *) &value;
        out.append(*x, 64);
        first = false;
        storedValue = value;
        size += 64;
    }

public:
    explicit CompressorChimp(const T& value) : size(0) {
        addValue(value);
    }

    inline void addValue(const T &value) {
        if (first) {
            writeFirst(value);
        } else {
            compressValue(value);
        }
    }

    void close() {
        //padding
        out.append(0, 64);
        out.close();
    }

    void compressValue(const T& value) {
        auto x = storedValue;
        auto y = value;
        auto *a = (uint64_t *) &x;
        auto *b = (uint64_t *) &y;
        uint64_t xor_ = *a ^ *b;

        if (xor_ == 0) {
            // Write 0
            out.push_back(0);
            out.push_back(0);
            size += 2;
            storedLeadingZeros = 65;
        } else {
            uint32_t leadingZeros = leadingRound[__builtin_clzll(xor_)];
            uint32_t trailingZeros = __builtin_ctzll(xor_);

            if (trailingZeros > THRESHOLD) {
                uint32_t significantBits = 64 - leadingZeros - trailingZeros;
                out.push_back(0);
                out.push_back(1);
                out.append(leadingRepresentation[leadingZeros], 3);
                out.append(significantBits, 6);
                xor_ >>= trailingZeros;
                out.append(xor_, significantBits); // Store the meaningful bits of XOR
                size += 11 + significantBits;
                storedLeadingZeros = 65;
            } else if (leadingZeros == storedLeadingZeros) {
                out.push_back(1);
                out.push_back(0);
                uint32_t significantBits = 64 - leadingZeros;
                out.append(xor_, significantBits);
                size += 2 + significantBits;
            } else {
                storedLeadingZeros = leadingZeros;
                uint32_t significantBits = 64 - leadingZeros;
                out.push_back(1);
                out.push_back(1);
                out.append(leadingRepresentation[leadingZeros], 3);
                out.append(xor_, significantBits);
                size += 5 + significantBits;
            }
        }
        storedValue = value;
    }

    size_t getSize() {
        return size;
    }

    inline BitStream getBuffer() {
        return this->out;
    }
};
