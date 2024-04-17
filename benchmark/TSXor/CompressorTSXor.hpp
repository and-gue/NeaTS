#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include "../lib/Window.hpp"
#include "../lib/Zigzag.hpp"

template<typename T>
struct CompressorTSXor {
    uint countA = 0;
    uint countB = 0;
    uint countC = 0;
    uint countB_bytes = 0;

    Window window{};
    std::vector<uint8_t> bytes{};

    explicit CompressorTSXor(const T &value) {
        writeFirst(value);
    }

    void addValue(const T &value) {
        compressValue(value);
    }

    void writeFirst(const T &value) {
        uint64_t x = *((uint64_t *) &(value));
        append64(x);
        window.insert(x);
    }

    void compressValue(const double &value) {
        uint64_t val = *((uint64_t *) &value);

        if (window.contains(val)) {
            auto offset = window.getIndexOf(val);
            auto *bytes = (uint8_t *) &offset;
            append8(bytes[0]);
            countA++;
        } else {
            uint64_t candidate = window.getCandidate(val);
            uint64_t xor_ = candidate ^ val;

            int lead_zeros_bytes = (__builtin_clzll(xor_) / 8);
            int trail_zeros_bytes = (__builtin_ctzll(xor_) / 8);

            if ((lead_zeros_bytes + trail_zeros_bytes) > 1) {
                auto offset = window.getIndexOf(candidate);

                //WRITE 1
                offset |= 0x80;
                auto *bytes = (uint8_t *) &offset;
                append8(bytes[0]);

                auto xor_len_bytes = 8 - lead_zeros_bytes - trail_zeros_bytes;
                xor_ >>= (trail_zeros_bytes * 8);

                uint8_t head = (trail_zeros_bytes << 4) | xor_len_bytes;
                append8(head);

                auto *xor_bytes = (uint8_t *) &xor_;
                for (int i = (xor_len_bytes - 1); i >= 0; i--) {
                    append8(xor_bytes[i]);
                }

                countB++;
                countB_bytes += (2 + xor_len_bytes);
            } else {
                append8((uint8_t) 255);
                append64(val);

                countC++;
            }
        }
        uint64_t _val = *((uint64_t *)&value);
        window.insert(_val);
    }

    void append64(uint64_t x) {
        auto *b = (uint8_t *) &x;
        for (int i = 7; i >= 0; i--) {
            bytes.push_back(b[i]);
        }
    }

    void close() {}

    inline void append8(uint8_t x) {
        bytes.push_back(x);
    }

    auto getBuffer() {
        return bytes;
    }

    auto getSize() {
        return bytes.size() * 8;
    }
};
