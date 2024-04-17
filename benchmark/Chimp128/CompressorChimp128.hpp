#pragma once

#include <iostream>
#include <limits>
#include "../lib/BitStream.hpp"
#include <cmath>
#include <vector>

template<typename T>
struct CompressorChimp128 {

    uint64_t storedLeadingZeros = std::numeric_limits<uint64_t>::max();
    bool first = true;
    size_t size = 0;
    constexpr static uint64_t previousValues = 128;
    std::array<uint64_t, previousValues> storedValues{};
    constexpr static uint64_t previousValuesLog2 = 7;
    constexpr static uint64_t threshold = 6 + previousValuesLog2;
    const int32_t setLsb = static_cast<int32_t>(pow(2, threshold + 1)) - 1;

    constexpr static uint8_t leadingRepresentation[] = {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7
    };

    constexpr static uint8_t leadingRound[] = {
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

    uint64_t index = 0;
    uint64_t current = 0;
    uint64_t flagZeroSize;
    uint64_t flagOneSize;

    std::vector<uint64_t> indices = std::vector<uint64_t>(static_cast<uint64_t>(pow(2, threshold + 1)));

    explicit CompressorChimp128(const T& value) {
        size = 0;
        flagZeroSize = previousValuesLog2 + 2;
        flagOneSize = previousValuesLog2 + 11;
        addValue(value);
    }

    auto getBuffer() {
        return out;
    }

    void addValue(const T& value) {
        if (first) {
            writeFirst(*(uint64_t*) &value);
        } else {
            compressValue(*(uint64_t*) &value);
        }
    }

    void writeFirst(const uint64_t &value) {
        out.append(value, 64);
        first = false;
        storedValues[current] = value;

        indices[*(int32_t*) &value & setLsb] = index;
        size += 64;
    }

    void close() {
        //padding
        //out.append(0, 64);
        out.append(0, 64);
        out.close();
    }

    void compressValue(const uint64_t &value) {

        int32_t key = (*(int32_t*) &value) & setLsb;

        uint64_t previousIndex;
        uint32_t trailingZeros = 0;
        uint32_t currIndex = indices[key];
        uint64_t xor_;

        if ((index - currIndex) < previousValues) {
            uint64_t tempXor = value ^ storedValues[currIndex % previousValues];
            trailingZeros = __builtin_ctzll(tempXor);
            if (trailingZeros > threshold) {
                previousIndex = currIndex % previousValues;
                xor_ = tempXor;
            } else {
                previousIndex = index % previousValues;
                xor_ = storedValues[previousIndex] ^ value;
            }
        } else {
            previousIndex = index % previousValues;
            xor_ = storedValues[previousIndex] ^ value;
        }

        if (xor_ == 0) {
            out.append(previousIndex, flagZeroSize);
            size += flagZeroSize;
            storedLeadingZeros = 65;
        } else {
            uint32_t leadingZeros = leadingRound[__builtin_clzll(xor_)];

            if (trailingZeros > threshold) {
                uint32_t significantBits = 64 - leadingZeros - trailingZeros;

                uint64_t v = (512 * (previousValues + previousIndex) + 64 * leadingRepresentation[leadingZeros] +
                             significantBits);
                out.append(v, flagOneSize);
                out.append(xor_ >> trailingZeros, significantBits);
                size += significantBits + flagOneSize;
                storedLeadingZeros = 65;
            } else if (leadingZeros == storedLeadingZeros) {
                out.append(2, 2);
                uint32_t significantBits = 64 - leadingZeros;
                out.append(xor_, significantBits);
                size += 2 + significantBits;
            } else {
                storedLeadingZeros = leadingZeros;
                uint32_t significantBits = 64 - leadingZeros;
                out.append(24 + leadingRepresentation[leadingZeros], 5);
                out.append(xor_, significantBits);
                size += 5 + significantBits;
            }
        }
        current = (current + 1) % previousValues;
        storedValues[current] = value;
        index++;
        indices[key] = index;
    }

    size_t getSize() { return size; }
};
