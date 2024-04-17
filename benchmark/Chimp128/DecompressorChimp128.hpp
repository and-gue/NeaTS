#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "../lib/BitStream.hpp"

template<typename T>
struct DecompressorChimp128 {

    uint64_t storedLeadingZeros = 0;
    uint64_t storedTrailingZeros = 0;
    T storedValue = 0;
    std::array<T, 128> storedValues{};
    uint64_t current = 0;
    bool endOfStream = false;
    BitStream in;
    constexpr static uint64_t previousValues = 128;
    constexpr static uint64_t previousValuesLog2 = 7;
    uint64_t initialFill;

    size_t n = 0;
    size_t i = 0;

    constexpr static uint16_t leadingRepresentation[] = {0, 8, 12, 16, 18, 20, 22, 24};

    DecompressorChimp128(const BitStream &bs, const size_t &nlines) {
        in = bs;
        n = nlines;
        initialFill = previousValuesLog2 + 9;

        uint64_t read = in.get(64);
        auto *p = (T *) &read;
        storedValue = *p;
        storedValues[current] = storedValue;
        ++i;
        if (i > n) {
            endOfStream = true;
        }
    }

    bool hasNext() {
        if (!endOfStream)
            nextValue();
        return !endOfStream;
    }

    void nextValue() {
        // Read value
        uint64_t read = in.get(2);
        uint64_t flag = *(uint64_t *) &read;

        uint64_t *a;
        uint64_t *b;
        uint64_t value;
        uint64_t xor_;
        uint64_t fill;

        uint64_t index;
        uint64_t tmp;
        uint32_t significantBits;
        switch (flag) {
            case 3:
                read = in.get(3);
                storedLeadingZeros = leadingRepresentation[*(uint64_t *) &(read)];
                value = in.get(64 - storedLeadingZeros);

                a = (uint64_t *) &storedValue;
                b = (uint64_t *) &value;
                xor_ = *a ^ *b;
                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                } else {
                    auto *p = (T *) &xor_;
                    storedValue = *p;
                    current = (current + 1) % previousValues;
                    storedValues[current] = storedValue;
                }
                break;
            case 2:
                value = in.get(64 - storedLeadingZeros);
                a = (uint64_t *) &storedValue;
                b = (uint64_t *) &value;
                xor_ = *a ^ *b;
                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                } else {
                    auto *p = (T *) &xor_;
                    storedValue = *p;
                    current = (current + 1) % previousValues;
                    storedValues[current] = storedValue;
                }
                break;
            case 1:
                fill = initialFill;

                read = in.get(fill);
                tmp = *(uint64_t *) &read;
                index = tmp >> (fill -= previousValuesLog2) & (1 << previousValuesLog2) - 1;
                storedLeadingZeros = leadingRepresentation[tmp >> (fill -= 3) & (1 << 3) - 1];
                significantBits = tmp >> (fill -= 6) & (1 << 6) - 1;
                storedValue = storedValues[index];
                if (significantBits == 0) {
                    significantBits = 64;
                }
                storedTrailingZeros = 64 - significantBits - storedLeadingZeros;
                value = in.get(64 - storedLeadingZeros - storedTrailingZeros);
                value <<= storedTrailingZeros;
                a = (uint64_t *) &storedValue;
                b = (uint64_t *) &value;
                xor_ = *a ^ *b;
                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                } else {
                    auto * p = (T *) &xor_;
                    storedValue = *p;
                    current = (current + 1) % previousValues;
                    storedValues[current] = storedValue;
                }
                break;
            default:
                // else -> same value as before
                read = in.get(previousValuesLog2);
                storedValue = storedValues[*(uint64_t*) &read];
                current = (current + 1) % previousValues;
                storedValues[current] = storedValue;

                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                }
                break;
        }
    }

};