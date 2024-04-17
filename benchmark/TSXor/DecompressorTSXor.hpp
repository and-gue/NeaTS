#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include "../lib/Zigzag.hpp"
#include "../lib/BitStream.hpp"
#include "../lib/Window.hpp"

template<typename T>
struct DecompressorTSXor {
    Window cache;

    T storedValue;

    bool endOfStream = false;

    size_t n;
    size_t i = 0;

    std::vector<uint8_t> bytes;
    uint64_t current_idx = 0;

    DecompressorTSXor(const std::vector<uint8_t> &bts, size_t nlines) {
        n = nlines;
        bytes = bts;
        cache = Window();

        uint64_t read = readBytes(8);
        cache.insert(read);
        T p = (*(T *) &read);
        storedValue = p;
        ++i;
    }

    bool hasNext() {
        if (!endOfStream)
            nextValue();
        return !endOfStream;
    }

    void nextValue() {
        uint64_t final_val;
        uint64_t offset;
        uint64_t info;
        uint64_t trail_zeros_bytes;
        uint64_t xor_bytes;
        uint64_t xor_;
        uint64_t head;

        head = readBytes(1);

        if (head < 128) {
            final_val = cache.get(head);
        } else if (head == 255) {
            final_val = readBytes(8);
        } else {
            offset = head & (~((UINT64_MAX << 7)));
            info = readBytes(1);
            trail_zeros_bytes = info >> 4;
            xor_bytes = info & (~((UINT64_MAX << 4)));
            xor_ = readBytes(xor_bytes) << (8 * trail_zeros_bytes);
            final_val = xor_ ^ cache.get(offset);
        }
        cache.insert(final_val);
        T p = (*(T *) &final_val);
        storedValue = p;
        ++i;
        if (i > n) endOfStream = true;
    }

    inline uint64_t readBytes(size_t len) {
        uint64_t val = 0;
        for (int j = 0; j < len; j++) {
            val |= bytes[current_idx];
            current_idx++;
            if (j != (len - 1)) {
                val <<= 8;
            }
        }
        return val;
    }
};
