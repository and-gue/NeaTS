#ifndef INCLUDED_SDSL_RLE_VECTOR
#define INCLUDED_SDSL_RLE_VECTOR

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "sd_vector.hpp"

//! Namespace for the succinct data structure library
namespace sdsl
{

//-----------------------------------------------------------------------------

// Forward declarations.

template<uint64_t t_block_size = 64>
class rle_vector;

template<uint8_t t_b = 1, uint64_t t_block_size = 64>
class rank_support_rle;

template<uint64_t t_block_size = 64>
class select_support_rle;

//-----------------------------------------------------------------------------

//! A growable bit-packed integer array that consists of fixed-size blocks.
/*!
 * \tparam t_block_bytes  Number of bytes per block. Must be at least 16 and a multiple of 8.
 * \tparam t_element_bits Number of bits per element. Must be a factor of 64.
 */
template<uint64_t t_block_bytes, uint64_t t_element_bits>
class block_array
{
    public:
        typedef bit_vector::size_type size_type;
        typedef uint64_t value_type;

    private:
        std::vector<value_type> data;

        static_assert(t_block_bytes >= 16, "block_array: block size must be at least 16 bytes.");
        static_assert(t_block_bytes % 8 == 0, "block_array: block size must be a multiple of 8 bytes.");
        static_assert(t_element_bits > 0 && 64 % t_element_bits == 0, "block_array: element size must be a factor of 64 bits.");

    public:
        //! Returns the size of the array in elements.
        size_type size() const { return this->data.size() * (64 / t_element_bits); }

        //! Returns the value of element i.
        value_type operator[](size_type i) const {
            i *= t_element_bits;
            return (this->data[i / 64] >> (i & 63)) & bits::lo_set[t_element_bits];
        }

        //! Sets the value of element i.
        void set(size_type i, value_type value) {
            i *= t_element_bits;
            this->data[i / 64] &= ~(bits::lo_set[t_element_bits] << (i & 63));
            this->data[i / 64] |= (value & bits::lo_set[t_element_bits]) << (i & 63);
        }

        //! Adds a new block to the array.
        void add_block() {
            this->data.resize(this->data.size() + (t_block_bytes / sizeof(value_type)), 0);
        }

        //! Equality comparison.
        bool operator==(const block_array& another) const
        {
            return (this->data == another.data);
        }

        //! Inequality comparison.
        bool operator!=(const block_array& another) const
        {
            return !(this->operator==(another));
        }

        //! Serializes the data structure into the given ostream.
        size_type serialize(std::ostream& out, structure_tree_node* v = nullptr, std::string name = "") const
        {
            structure_tree_node* child = structure_tree::add_child(v, name, util::class_name(*this));
            size_type written_bytes = 0;

            size_type len = this->data.size();
            written_bytes += write_member(len, out);
            size_type offset = 0;
            while (offset < this->data.size()) {
                size_type words = std::min(this->data.size() - offset, conf::SDSL_BLOCK_SIZE / sizeof(uint64_t));
                out.write((const char*)(this->data.data() + offset), words * sizeof(uint64_t));
                written_bytes += words * sizeof(uint64_t);
                offset += words;
            }
 
            structure_tree::add_size(child, written_bytes);
            return written_bytes;
        }

        //! Loads the data structure from the given istream.
        void load(std::istream& in)
        {
            size_type len = 0;
            read_member(len, in);
            this->data.resize(len, 0);

            size_type offset = 0;
            while (offset < this->data.size()) {
                size_type words = std::min(this->data.size() - offset, conf::SDSL_BLOCK_SIZE / sizeof(uint64_t));
                in.read((char*)(this->data.data() + offset), words * sizeof(uint64_t));
                offset += words;
            }
        }

        //! Swap method.
        void swap(block_array& another) {
            this->data.swap(another.data);
        }
};

//-----------------------------------------------------------------------------

//! Class for in-place construction of rle_vector from a strictly increasing sequence.
/*!
 * \tparam t_block_size Number of elements in a block. Must be a multiple of 32.
 * \par Building a `rle_vector` will clear the builder.
 */
template<uint64_t t_block_size = 64>
class rle_vector_builder
{
    public:
        typedef sd_vector<>::size_type size_type;

    private:
        size_type run_start, run_length;

        size_type length;    // Final length of the bitvector.
        size_type total_bits, set_bits;

        block_array<t_block_size / 2, 4> body;
        size_type body_tail; // Next unused position in the body.

        std::vector<size_type> block_bits, block_ones;

        constexpr static size_type DATA_MASK = 0x7;
        constexpr static size_type NEXT_ELEM = 0x8;
        constexpr static size_type DATA_BITS = 3;

        static_assert(t_block_size >= 32 && t_block_size % 32 == 0, "rle_vector_builder: block size must be a positive multiple of 32.");

    public:
        //! Creates a builder for a rle_vector of length n.
        explicit rle_vector_builder(size_type n = 0) :
            run_start(0), run_length(0),
            length(n), total_bits(0), set_bits(0),
            body_tail(0)
        {
        }

        //! Returns the length of the bitvector.
        size_type size() const { return this->length; }

        //! Returns the first bitvector position that can be set.
        size_type tail() const { return this->run_start + this->run_length; }

        //! Appends one or more 1s to the bitvector.
        /*! \param i The position of the first 1.
         *  \param n Number of bits to append.
         *  \par The position must be strictly greater than the last set bit.
         *  Behavior is undefined if the position is out of range or if `n == 0`.
         */
        void set_unsafe(size_type i, size_type n = 1) noexcept
        {
            if (i == this->tail()) { this->run_length += n; }
            else {
                this->write_run();
                this->run_start = i; this->run_length = n;
            }
        }

        //! Appends one or more 1s to the bitvector.
        /*! \param i The position of the first 1.
         *  \param n Number of bits to append.
         *  \par The position must be strictly greater than the last set bit.
         *  Throws `std::runtime_error` if the position is out of range.
         */
        void set(size_type i, size_type n = 1)
        {
            if (n == 0) { return; }
            if (i < this->tail()) {
                throw std::runtime_error("rle_vector_builder::set(): the position is too small.");
            }
            if (i + n > this->size()) {
                throw std::runtime_error("sd_vector_builder::set(): the position is too large.");
            }
            this->set_unsafe(i, n);
        }

    private:
        template<uint64_t> friend class rle_vector;

        void add_block() {
            this->body_tail = this->body.size();
            this->body.add_block();
            this->block_bits.push_back(this->total_bits);
            this->block_ones.push_back(this->set_bits);
        }

        // Is there enough space in the current block to encode `value`?
        bool fits_in_current_block(size_type value) {
            size_type needed = (bits::length(value) + DATA_BITS - 1) / DATA_BITS;
            return (this->body.size() - this->body_tail >= needed);
        }

        // Append the value to the body, assuming that it fits into the current block.
        void write(size_type value) {
            while (value > DATA_MASK) {
                this->body.set(this->body_tail, (value & DATA_MASK) | NEXT_ELEM);
                this->body_tail++; value >>= DATA_BITS;
            }
            this->body.set(this->body_tail, value);
            this->body_tail++;
        }

        // Add 0s until position `i`. Note that this creates a new run of 0s.
        void add_zeros_until(size_type i) {
            size_type zero_run = i - this->total_bits;
            if (!this->fits_in_current_block(zero_run)) {
                this->add_block();
            }
            this->write(zero_run);
            this->total_bits = i;
        }

        // Append the current run to the bitvector if its length is nonzero.
        void write_run() {
            if (this->run_length == 0) { return; }

            // Write the run of 0s.
            this->add_zeros_until(this->run_start);

            // Write the run of 1s.
            if (!this->fits_in_current_block(this->run_length - 1)) {
                this->add_block();
                this->write(0); // A block always starts with a run of 0s.
            }
            this->write(this->run_length - 1);
            this->total_bits += this->run_length;
            this->set_bits += this->run_length;
        }

        // Finish the construction. Further bits cannot be set after this.
        void flush() {
            this->write_run();
            if (this->total_bits < this->size()) { this->add_zeros_until(this->size()); }
        }
};

//-----------------------------------------------------------------------------

//! A bitvector that uses run-length encoding.
/*!
 * \par The overall design is related to the bitvectors used in RLCSA.
 * The body of the bitvector consists of an alternating sequence of runs of 0s and 1s.
 * Run lengths are encoded using elements of 4 bits (3 bits of data + carry bit).
 * Runs of 0s are encoded as is, while runs of 1s are encoded as (length - 1).
 * The runs are divided into blocks of `t_block_size` elements (default 64).
 * Two `sd_vector`s are used for storing the number of bits and 1s before each block.
 *
 * \par References
 *  - V. Mäkinen, G. Navarro, J. Sirén, N. Välimäki:
 *    ,, Storage and Retrieval of Highly Repetitive Sequence Collections'',
 *    Journal of Computational Biology, 2010.
 * 
 * \tparam t_block_size Number of elements in a block. Must be a multiple of 32.
*/
template<uint64_t t_block_size>
class rle_vector
{
    // Definitions.

    public:
        typedef bit_vector::size_type                    size_type;
        typedef size_type                                value_type;
        typedef bit_vector::difference_type              difference_type;
        typedef random_access_const_iterator<rle_vector> iterator;
        typedef iterator                                 const_iterator;
        typedef bv_tag                                   index_category;

        typedef rank_support_rle<0, t_block_size>        rank_0_type;
        typedef rank_support_rle<1, t_block_size>        rank_1_type;
        typedef select_support_rle<t_block_size>         select_1_type;

        typedef rle_vector_builder<t_block_size>         builder_type;

    private:
        block_array<t_block_size / 2, 4> body;
        sd_vector<> block_bits; // Marks the first bit in each block.
        sd_vector<> block_ones; // Marks the first one in each block.

        constexpr static size_type DATA_MASK = 0x7;
        constexpr static size_type NEXT_ELEM = 0x8;
        constexpr static size_type DATA_BITS = 3;

        static_assert(t_block_size >= 32 && t_block_size % 32 == 0, "rle_vector: block size must be a positive multiple of 32.");

//-----------------------------------------------------------------------------

    // Standard interface.

    public:
        rle_vector() {}
        rle_vector(const rle_vector& source) { this->copy(source); }
        rle_vector(rle_vector&& source) { *this = std::move(source); }

        rle_vector& operator=(const rle_vector& source)
        {
            if (this != &source) { this->copy(source); }
            return *this;
        }

        rle_vector& operator=(rle_vector&& source)
        {
            if (this != &source) {
                this->body = std::move(source.body);
                this->block_bits = std::move(source.block_bits);
                this->block_ones = std::move(source.block_ones);
            }
            return *this;
        }

        //! Swap method
        void swap(rle_vector& another)
        {
            if (this != &another) {
                this->body.swap(another.body);
                this->block_bits.swap(another.block_bits);
                this->block_ones.swap(another.block_ones);
            }
        }

        //! Equality comparison.
        bool operator==(const rle_vector& another) const
        {
            return (this->body == another.body && this->block_bits == another.block_bits && this->block_ones == another.block_ones);
        }

        //! Inequality comparison.
        bool operator!=(const rle_vector& another) const
        {
            return !(this->operator==(another));
        }

        //! Serializes the data structure into the given ostream.
        size_type serialize(std::ostream& out, structure_tree_node* v = nullptr, std::string name = "") const
        {
            structure_tree_node* child = structure_tree::add_child(v, name, util::class_name(*this));
            size_type written_bytes = 0;
            written_bytes += this->body.serialize(out, child, "body");
            written_bytes += this->block_bits.serialize(out, child, "block_bits");
            written_bytes += this->block_ones.serialize(out, child, "block_ones");
            structure_tree::add_size(child, written_bytes);
            return written_bytes;
        }

        //! Loads the data structure from the given istream.
        void load(std::istream& in)
        {
            this->body.load(in);
            this->block_bits.load(in);
            this->block_ones.load(in);
        }

//-----------------------------------------------------------------------------

    // Main constructors.

    public:
        //! Create a run-length encoded copy of the source bitvector.
        rle_vector(const bit_vector& source)
        {
            builder_type builder(source.size());
            for (size_type i = 0; i < source.size(); i++) {
                if (source[i]) { builder.set_unsafe(i); }
            }
            *this = rle_vector(builder);
        }

        //! Convert the data in the builder into a `rle_vector`.
        /*!
         * \par Construction clears the builder.
         */
        rle_vector(builder_type& builder)
        {
            builder.flush();
            this->body = std::move(builder.body);

            if (builder.total_bits > 0) {
                sd_vector_builder bits_builder(builder.total_bits, builder.block_bits.size());
                for (auto pos : builder.block_bits) { bits_builder.set_unsafe(pos); }
                this->block_bits = sd_vector<>(bits_builder);
            }

            if (builder.set_bits > 0) {
                sd_vector_builder ones_builder(builder.set_bits, builder.block_ones.size());
                for (auto pos : builder.block_ones) { ones_builder.set_unsafe(pos); }
                this->block_ones = sd_vector<>(ones_builder);
            }

            builder = builder_type();
        }

//-----------------------------------------------------------------------------

    // Operations.

    public:
        //! Returns the i-th element of the bitvector.
        /*! \param i Position in the bitvector.
         */
        value_type operator[](size_type i) const
        {
            if (i >= this->size()) { return 0; }

            auto sample = *(this->block_bits.predecessor(i));
            size_type body_offset = sample.first * t_block_size;
            size_type bv_offset = sample.second;
            while (true) {
                bv_offset += this->run_of_zeros(body_offset);
                if (bv_offset > i) { return 0; }
                bv_offset += this->run_of_ones(body_offset);
                if (bv_offset > i) { return 1; }
            }
        }

        //! Returns the integer value of the binary string of length len starting at position i.
        /*!
         * \param i   Starting position in the bitvector.
         * \param len Length of the binary string (1 to 64).
         * \par Behavior is undefined if `len` is invalid.
         */
        uint64_t get_int(size_type i, size_type len = 64) const
        {
            if (i >= this->size()) { return 0; }
            len = std::min(len, this->size() - i);

            auto iter = this->block_bits.predecessor(i);
            size_type bv_offset = iter->second;
            uint64_t value = 0;

            // Process all the necessary blocks.
            while (bv_offset < i + len) {
                size_type body_offset = iter->first * t_block_size;
                ++iter;
                bool need_next_block = (iter != this->block_bits.one_end() && iter->second < i + len);
                size_type block_limit = (need_next_block ? iter->second : i + len);

                // Process the current block.
                while (true) {
                    bv_offset += this->run_of_zeros(body_offset);
                    if (bv_offset >= block_limit) { break; }
                    size_type one_run = this->run_of_ones(body_offset);
                    if (bv_offset + one_run > i) {
                        if (bv_offset < i) {
                            value = bits::lo_set[std::min(one_run - (i - bv_offset), len)];
                        } else {
                            value|= bits::lo_set[std::min(one_run, block_limit - bv_offset)] << (bv_offset - i);
                        }
                    }
                    bv_offset += one_run;
                    if (bv_offset >= block_limit) { break; }
                }
            }

            return value;
        }

        // TODO other operations?

//-----------------------------------------------------------------------------

    // Iterators and statistics.

    public:
        iterator begin() const { return iterator(this, 0); }
        iterator end() const { return iterator(this, this->size()); }

        //! Returns the length of the bitvector.
        size_type size() const { return this->block_bits.size(); }

        //! Returns the number of ones in the bitvector.
        size_type ones() const { return this->block_ones.size(); }

        //! Counts the number of runs of 1s in the bitvector.
        /*!
         * \par This call is somewhat expensive, because it has to iterate over the runs.
         */
        size_type runs() const
        {
            size_type result = 0;

            size_type found_ones = 0;
            auto iter = this->block_ones.one_begin();
            while (iter != this->block_ones.one_end()) {
                size_type body_offset = iter->first * t_block_size;
                ++iter;
                while (found_ones < iter->second) {
                    this->run_of_zeros(body_offset);
                    found_ones += this->run_of_ones(body_offset);
                    result++;
                }
            }

            return result;
        }

//-----------------------------------------------------------------------------

    // Internal implementation.

    private:
        template<uint8_t, uint64_t> friend class rank_support_rle;
        friend class select_support_rle<t_block_size>;

        void copy(const rle_vector& another)
        {
            this->body = another.body;
            this->block_bits = another.block_bits;
            this->block_ones = another.block_ones;
        }

        // Returns the (raw) length of the run starting at `body[i]`.
        // The actual length of a run of 1s is `run_length() + 1`.
        size_type run_length(size_type& i) const
        {
            size_type offset = 0;
            size_type result = this->body[i] & DATA_MASK;
            while ((this->body[i] & NEXT_ELEM) != 0) {
                i++; offset += DATA_BITS;
                result += (this->body[i] & DATA_MASK) << offset;
            }
            i++;
            return result;
        }

        size_type run_of_zeros(size_type& i) const { return this->run_length(i); }
        size_type run_of_ones(size_type& i) const { return this->run_length(i) + 1; }
};

//-----------------------------------------------------------------------------

template<uint8_t t_b>
struct rank_support_rle_trait
{
    typedef rle_vector<>::size_type size_type;
    static size_type adjust_rank(size_type r, size_type) { return r; }
};

template<>
struct rank_support_rle_trait<0>
{
    typedef rle_vector<>::size_type size_type;
    static size_type adjust_rank(size_type r, size_type n) { return n - r; }
};

//! Rank data structure for rle_vector.
/*! \tparam t_b          Bit pattern.
 *  \tparam t_block_size Block size in the `rle_vector`.
 */
template<uint8_t t_b, uint64_t t_block_size>
class rank_support_rle
{
    public:
        typedef rle_vector<t_block_size> bit_vector_type;
        typedef typename bit_vector_type::size_type size_type;
        enum { bit_pat = t_b };
        enum { bit_pat_len = (uint8_t)1 };

        explicit rank_support_rle(const bit_vector_type* v = nullptr) : parent(v) {}

        //! Returns the number of bits of type t_b before position i.
        /*!
         * \param i Position in the bitvector.
         * \par Returns the total number of bits of type `t_b` if `i` is too large.
         */
        size_type rank(size_type i) const
        {
            if (i >= this->parent->size()) { return rank_support_rle_trait<t_b>::adjust_rank(this->parent->ones(), this->parent->size()); }

            auto sample = *(this->parent->block_bits.predecessor(i));
            size_type body_offset = sample.first * t_block_size;
            size_type bv_offset = sample.second;
            size_type result = this->parent->block_ones.select_iter(sample.first + 1)->second;
            while (true) {
                bv_offset += this->parent->run_of_zeros(body_offset);
                if (bv_offset >= i) { break; }
                size_type one_run = this->parent->run_of_ones(body_offset);
                bv_offset += one_run; result += one_run;
                if (bv_offset >= i) {
                    result -= bv_offset - i;
                    break;
                }
            }

            return rank_support_rle_trait<t_b>::adjust_rank(result, i);
        }

        size_type operator()(size_type i) const { return this->rank(i); }

        void set_vector(const bit_vector_type* v = nullptr) { this->parent = v; }

        rank_support_rle& operator=(const rank_support_rle& another)
        {
            if (this != &another) { this->set_vector(another.parent); }
            return *this;
        }

        void swap(rank_support_rle& another) { std::swap(this->parent, another.parent); }

        void load(std::istream&, const bit_vector_type* v = nullptr) { this->set_vector(v); }

        size_type serialize(std::ostream& out, structure_tree_node* v = nullptr, std::string name = "") const
        {
            return serialize_empty_object(out, v, name, this);
        }

    private:
        static_assert(t_b == 1u or t_b == 0u , "rank_support_rle: bit pattern must be `0` or `1`");
        const bit_vector_type* parent;
};

//-----------------------------------------------------------------------------


//! Select data structure for rle_vector.
/*! \tparam t_block_size Block size in the `rle_vector`.
 */
template<uint64_t t_block_size>
class select_support_rle
{
    public:
        typedef rle_vector<t_block_size> bit_vector_type;
        typedef typename bit_vector_type::size_type size_type;
        enum { bit_pat = (uint8_t)1 };
        enum { bit_pat_len = (uint8_t)1 };

        explicit select_support_rle(const bit_vector_type* v = nullptr) : parent(v) {}

        //! Returns the position of the i-th one in the bitvector.
        /*!
         * \param i 1-based rank of the bit.
         * \par Returns -1 if `i == 0` and the length of the bitvector if `i` is too large.
         */
        size_type select(size_type i) const
        {
            if (i == 0) { return static_cast<size_type>(-1); }
            if (i > this->parent->ones()) { return this->parent->size(); }

            auto sample = *(this->parent->block_ones.predecessor(i - 1)); // select(i) corresponds to block_ones[i - 1].
            size_type body_offset = sample.first * t_block_size;
            size_type bv_offset = this->parent->block_bits.select_iter(sample.first + 1)->second;
            size_type rank = sample.second;
            while (true) {
                bv_offset += this->parent->run_of_zeros(body_offset);
                size_type one_run = this->parent->run_of_ones(body_offset);
                bv_offset += one_run; rank += one_run;
                if (rank >= i) {
                    return bv_offset - 1 - (rank - i);
                }
            }
        }

        size_type operator()(size_type i) const { return this->select(i); }

        void set_vector(const bit_vector_type* v = nullptr) { this->parent = v; }

        select_support_rle& operator=(const select_support_rle& another)
        {
            if (this != &another) { this->set_vector(another.parent); }
            return *this;
        }

        void swap(select_support_rle& another) { std::swap(this->parent, another.parent); }

        void load(std::istream&, const bit_vector_type* v = nullptr) { this->set_vector(v); }

        size_type serialize(std::ostream& out, structure_tree_node* v = nullptr, std::string name = "") const
        {
            return serialize_empty_object(out, v, name, this);
        }

    private:
        const bit_vector_type* parent;
};

//-----------------------------------------------------------------------------

} // namespace sdsl

#endif // INCLUDED_SDSL_RLE_VECTOR