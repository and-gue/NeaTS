#pragma once

#include <iostream>
#include <limits>
#include "../lib/BitStream.hpp"
#include <vector>

template<typename T>
struct DecompressorChimp {

    uint64_t storedLeadingZeros = 0;
    uint64_t storedTrailingZeros = 0;
    T storedValue = 0;
    bool endOfStream = false;

    BitStream in;
    constexpr static uint16_t leadingRepresentation[] = {0, 8, 12, 16, 18, 20, 22, 24};
    size_t n;
    size_t i;

    explicit DecompressorChimp(const BitStream &bs, const size_t &nlines) : i(0), n(nlines) {
        in = bs;

        uint64_t read = in.get(64);
        T *p = (T *) &read;
        storedValue = *p;
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
        uint32_t significantBits;
        uint64_t xor_;
        // Read value
        uint64_t read = in.get(2);
        uint64_t flag = *(uint64_t *) &read;

        uint64_t *a;
        uint64_t *b;
        uint64_t value;

        switch (flag) {
            case 3:
                // New leading zeros
                read = in.get(3);
                storedLeadingZeros = leadingRepresentation[*(uint64_t *) &(read)];
                significantBits = 64 - storedLeadingZeros;
                if (significantBits == 0) {
                    significantBits = 64;
                }

                value = in.get(64 - storedLeadingZeros);
                a = (uint64_t *) &storedValue;
                b = (uint64_t *) &value;
                xor_ = *a ^ *b;

                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                } else {
                    T *p = (T *) &xor_;
                    storedValue = *p;
                }
                break;
            case 2:
                significantBits = 64 - storedLeadingZeros;
                if (significantBits == 0) {
                    significantBits = 64;
                }

                value = in.get(64 - storedLeadingZeros);
                a = (uint64_t *) &storedValue;
                b = (uint64_t *) &value;
                xor_ = *a ^ *b;

                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                } else {
                    T *p = (T *) &xor_;
                    storedValue = *p;
                }
                break;
            case 1:
                read = in.get(3);
                storedLeadingZeros = leadingRepresentation[*(uint64_t *) &(read)];

                read = in.get(6);
                significantBits = *(uint32_t *) &read;

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
                    T *p = (T *) &xor_;
                    storedValue = *p;
                }
                break;
            default:
                ++i;
                if (i > n) {
                    endOfStream = true;
                    return;
                }
                break;
        }
    }

};