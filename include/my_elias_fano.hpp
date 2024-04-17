//
// Created by Giorgio Vinciguerra on 26/03/23.
//

#pragma once

#include <cstdint>
#include <sdsl/int_vector.hpp>
#include <sux/bits/SimpleSelect.hpp>
#include <sux/bits/SimpleSelectHalf.hpp>
#include <sux/bits/SimpleSelectZero.hpp>
#include <sux/bits/SimpleSelectZeroHalf.hpp>

// Returns the width of the integers stored in the low part of an Elias-Fano-coded sequence.
//
// This is based on:
//
//   Ma, Puglisi, Raman, Zhukova:
//   On Elias-Fano for Rank Queries in FM-Indexes.
//   DCC 2021.
//
// Implementation credit: Jouni Siren, https://github.com/vgteam/sdsl-lite
static uint8_t elias_fano_lo_width(size_t universe, size_t ones) {
    uint8_t low_width = 1;
    // Multisets with too many ones will have width 1.
    if (ones > 0 && ones <= universe) {
        double ideal_width = std::log2((static_cast<double>(universe) * std::log(2.0)) / static_cast<double>(ones));
        low_width = (uint8_t) std::round(std::max(ideal_width, 1.0));
    }
    return low_width;
}

/** Finds the next set bit after a given position. */
inline uint64_t next_one(size_t i, const uint64_t *data) {
    auto j = i / 64;
    auto word = data[j] & sdsl::bits::lo_unset[i % 64 + 1];
    while (word == 0)
        word = data[++j];
    return j * 64 + __builtin_ctzll(word);
}

/** Finds the previous set bit before a given position. */
inline uint64_t prev_one(size_t i, const uint64_t *data) {
    auto j = i / 64;
    auto word = data[j] & sdsl::bits::lo_set[i % 64];
    while (word == 0)
        word = data[--j];
    return j * 64 + 63 - __builtin_clzll(word);
}

/** Finds the next zero bit after a given position. */
inline uint64_t next_zero(size_t i, const uint64_t *data) {
    auto j = i / 64;
    auto word = data[j] | sdsl::bits::lo_set[i % 64 + 1];
    while (word == std::numeric_limits<uint64_t>::max())
        word = data[++j];
    return j * 64 + __builtin_ctzll(~word);
}

template<bool AllowRank = true>
class MyEliasFano {
    sdsl::int_vector<> v;
    sdsl::bit_vector H;
    uint8_t lo_width;
    size_t n = 0;
    uint64_t u = 0;
    bool large_bucket = false;
    sux::bits::SimpleSelectHalf<> select1;
    sux::bits::SimpleSelectZeroHalf<> select0;

    class Iterator;

public:

    MyEliasFano() = default;

    explicit MyEliasFano(const std::vector<uint64_t> &data) {
        if (data.empty())
            return;

        u = data.back() + 1;
        n = data.size();

        lo_width = elias_fano_lo_width(u, n);
        v.width(lo_width);
        v.resize(n);

        H = decltype(H)((u >> lo_width) + n + 1, 0);

        size_t prev_bucket = 0;
        size_t max_bucket_size = 0;
        size_t bucket_size = 0;
        size_t non_empty_buckets = 0;

        for (size_t i = 0; i < data.size(); i++) {
            if (lo_width > 0)
                v[i] = data[i] & mask();
            auto bucket = data[i] >> lo_width;
            H[bucket + i] = true;
            if (bucket == prev_bucket) {
                bucket_size++;
            } else {
                max_bucket_size = std::max(max_bucket_size, bucket_size);
                bucket_size = 1;
                prev_bucket = bucket;
                non_empty_buckets++;
            }
        }

        large_bucket = max_bucket_size > 512;

        select1 = {H.data(), H.size()};
        if constexpr (AllowRank)
            select0 = {H.data(), H.size()};
    }

    /** Returns the largest value in the sequence that is <= x. */
    [[nodiscard]] Iterator predecessor(uint64_t x) const {
        static_assert(AllowRank, "Cannot call predecessor() if AllowRank is false");
        if (x >= u - 1)
            return at(n - 1);

        const uint64_t x_upper = x >> lo_width;
        const uint64_t x_lower = x & mask();

        size_t pos_hi;
        size_t pos_lo = 0;
        if (x_upper == 0) {
            pos_hi = select0.selectZero(x_upper);
        } else if (large_bucket) {
            pos_lo = select0.selectZero(x_upper - 1) + 1;
            pos_hi = select0.selectZero(x_upper);
        } else {
            pos_lo = select0.selectZero(x_upper - 1);
            pos_hi = next_zero(pos_lo, H.data());
            ++pos_lo;
        }

        const auto rank_lo = pos_lo - x_upper;
        const auto rank_hi = pos_hi - x_upper;
        auto count = pos_hi - pos_lo;
        size_t pos;
        size_t rank;

        if (count < 8) {
            rank = rank_hi;
            do {
                --rank;
            } while (rank >= rank_lo && v[rank] > x_lower);
        } else {
            rank = rank_lo;
            while (count > 0) {
                auto step = count / 2;
                auto mid = rank + step;
                if (v[mid] <= x_lower) {
                    rank = mid + 1;
                    count -= step + 1;
                } else {
                    count = step;
                }
            }
            --rank;
        }

        pos = pos_hi - (rank_hi - rank);

        if (pos < pos_lo)
            pos = prev_one(pos, H.data());

        return {rank, pos, this};
    }

    [[nodiscard]] uint64_t operator[](size_t rank) const {
        uint64_t l = lo_width ? v[rank] : 0;
        auto pos = select1.select(rank);
        return ((pos - rank) << lo_width) | l;
    }

    [[nodiscard]] Iterator at(size_t rank) const {
        return {rank, select1.select(rank), this};
    }

    [[nodiscard]] uint64_t universe_size() const { return u; }

    [[nodiscard]] size_t size() const { return n; }

    [[nodiscard]] size_t size_in_bytes() const {
        return sdsl::size_in_bytes(v) + sdsl::size_in_bytes(H) + select0.bitCount() / 8 + select1.bitCount() / 8;
    }

    size_t inline serialize(std::ostream &os, sdsl::structure_tree_node *_v = nullptr, std::string name = "") const {
        size_t written_bytes = 0;
        written_bytes += sdsl::write_member(AllowRank, os, _v, name + "_AllowRank");
        written_bytes += sdsl::write_member(u, os, _v, name + "_u");
        written_bytes += sdsl::write_member(n, os, _v, name + "_n");
        written_bytes += sdsl::write_member(lo_width, os, _v, name + "_lo_width");
        written_bytes += sdsl::serialize(v, os, _v, name + "_v");
        written_bytes += sdsl::serialize(H, os, _v, name + "_H");
        //written_bytes += select1.serialize(os, v, name + "_select1");
        //written_bytes += select0.serialize(os, v, name + "_select0");
        return written_bytes;
    }

    void inline load(std::istream &is) {
        bool _AllowRank;
        sdsl::read_member(_AllowRank, is);
        if (_AllowRank != AllowRank)
            throw std::runtime_error("AllowRank mismatch");
        sdsl::read_member(u, is);
        sdsl::read_member(n, is);
        sdsl::read_member(lo_width, is);
        sdsl::load(v, is);
        sdsl::load(H, is);
        //select1.load(is, v);
        //select0.load(is, v);
        select1 = decltype(select1){H.data(), H.size()};
        if constexpr (AllowRank)
            select0 = decltype(select0){H.data(), H.size()};
    }

private:

    [[nodiscard]] uint64_t mask() const { return sdsl::bits::lo_set[lo_width]; }

    class Iterator {
        size_t rank;
        size_t pos;
        const MyEliasFano *ef;

    public:

        Iterator(size_t rank, size_t pos, const MyEliasFano *ef) : rank(rank), pos(pos), ef(ef) {}

        Iterator &operator++() {
            rank++;
            pos = next_one(pos, ef->H.data());
            return *this;
        }

        Iterator &operator--() {
            rank--;
            pos = prev_one(pos, ef->H.data());
            return *this;
        }

        uint64_t operator*() const {
            uint64_t l = ef->lo_width ? ef->v[rank] : 0;
            return ((pos - rank) << ef->lo_width) | l;
        }

        [[nodiscard]] size_t index() const {
            return rank;
        }
    };
};