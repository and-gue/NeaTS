#include <vector>
#include <iostream>
#include <string>
#include "../lib/BitStream.hpp"
#include "../lib/Zigzag.hpp"

template<typename T>
struct DecompressorGorilla {
    static_assert(std::same_as<T, double> || std::same_as<T, uint64_t>);

    uint64_t storedLeadingZeros;
    uint64_t storedTrailingZeros;
    T storedValue;

    BitStream bs_values{};
    bool endOfStream = false;

    size_t n;
    size_t i;

    DecompressorGorilla(const BitStream &bs_val, size_t nlines) {
        bs_values = bs_val;
        storedLeadingZeros = 0;
        storedTrailingZeros = 0;
        i = 0;
        n = nlines;

        uint64_t read = bs_values.get(64);
        T *p = (T *) &read;
        storedValue = *p;
        ++i;
    }

    bool hasNext() {
        if (!endOfStream)
            nextValue();
        return !endOfStream;
    }

    void nextValue() {
        // Read value
        // If 1 means that the value has not changed, hence no ops perfomed
        if (bs_values.readBit()) {
            // else -> same value as before
            if (bs_values.readBit()) {
                // New leading and trailing zeros
                storedLeadingZeros = bs_values.get(5);

                uint64_t significantBits = bs_values.get(6);
                if (significantBits == 0) {
                    significantBits = 64;
                }
                storedTrailingZeros = 64 - significantBits - storedLeadingZeros;
            }
            uint64_t value = bs_values.get(64 - storedLeadingZeros - storedTrailingZeros);
            value <<= storedTrailingZeros;

            uint64_t *a = (uint64_t *) &storedValue;
            uint64_t *b = (uint64_t *) &value;
            uint64_t xor_ = *a ^ *b;
            T *p = (T *) &xor_;
            storedValue = *p;
        }
        ++i;
        if (i > n) endOfStream = true;
    }

};
