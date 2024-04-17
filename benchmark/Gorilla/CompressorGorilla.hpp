#include <vector>
#include <map>
#include <string>
#include "../lib/Zigzag.hpp"
#include "../lib/BitStream.hpp"

template<typename T>
struct CompressorGorilla {
    static_assert(std::same_as<T, double> || std::same_as<T, uint64_t>);

    uint64_t storedLeadingZeros;
    uint64_t storedTrailingZeros;
    T storedValue;

    BitStream bs_values{};

    explicit CompressorGorilla(const T &value) {
        auto *x = (uint64_t *) &value;
        bs_values.append(*x, 64);
        storedValue = value;
        storedLeadingZeros = 0;
        storedTrailingZeros = 64;
    }

    inline void addValue(const T &val) {
        compressValue(val);
    }

    void close() {
        //padding
        bs_values.append(0, 64);
        bs_values.close();
    }

    void compressValue(const T &value) {
        auto x = storedValue;
        auto y = value;
        auto *a = (uint64_t *) &x;
        auto *b = (uint64_t *) &y;
        uint64_t xor_ = *a ^ *b;

        if (xor_ == 0) {
            // Write 0
            bs_values.push_back(0);
            // countA++;
        } else {
            int leadingZeros = __builtin_clzll(xor_);
            int trailingZeros = __builtin_ctzll(xor_);

            if (leadingZeros >= 32) {
                leadingZeros = 31;
            }

            if (leadingZeros == trailingZeros) {
                xor_ = xor_ >> 1 << 1;
                trailingZeros = 1;
            }

            // Store bit '1'
            bs_values.push_back(1);

            if (leadingZeros >= storedLeadingZeros && trailingZeros >= storedTrailingZeros) {
                bs_values.push_back(0);
                int significantBits = 64 - storedLeadingZeros - storedTrailingZeros;
                xor_ >>= storedTrailingZeros;
                bs_values.append(xor_, significantBits);
            } else {
                int significantBits = 64 - leadingZeros - trailingZeros;

                bs_values.append((((0x20 ^ leadingZeros) << 6) ^ (significantBits)), 12);
                xor_ >>= trailingZeros;                  // Length of meaningful bits in the next 6 bits
                bs_values.append(xor_, significantBits); // Store the meaningful bits of XOR

                storedLeadingZeros = leadingZeros;
                storedTrailingZeros = trailingZeros;
            }
        }
        storedValue = value;
    }

    size_t getSize() {
        return bs_values.size();
    }

    auto getBuffer() {
        return bs_values;
    }
};
