// Copyright (c) 2016, the SDSL Project Authors.  All rights reserved.
// Please see the AUTHORS file for details.  Use of this source code is governed
// by a BSD license that can be found in the LICENSE file.
/*!\file select_support_mcl.hpp
 * \brief select_support_mcl.hpp contains classes that support a sdsl::bit_vector with constant time select information.
 * \author Simon Gog
 */
#ifndef INCLUDED_SDSL_SELECT_SUPPORT_MCL
#define INCLUDED_SDSL_SELECT_SUPPORT_MCL

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <string>

#include <sdsl/bits.hpp>
#include <sdsl/cereal.hpp>
#include <sdsl/int_vector.hpp>
#include <sdsl/select_support.hpp>
#include <sdsl/structure_tree.hpp>
#include <sdsl/util.hpp>

//! Namespace for the succinct data structure library.
namespace sdsl
{

//! A class supporting constant time select queries.
/*!
 * \par Space usage
 *      The space usage of the data structure depends on the number of \f$ m \f$ of ones in the
 *      original bitvector $b$. We store the position of every $4096$th set bit
 *      (called L1-sampled bits) of $b$.
 *      This takes in the worst case \f$\frac{m}{4096} \log{n} \leq \frac{n}{64}\f$ bits.
 *      Next,
 *      (1) if the distance of two adjacent L1-sampled bits $b[i]$ and $b[j]$
 *      is greater or equal than $\log^4 n$, then
 *      we store each of the 4096 positions of the set $b$ in [i..j-1] with
 *      $\log{n}$ bits. This results in at most
 *      \$ \frac{4096\cdot \log n}{\log^4 n}=\frac{4096}{\log^3 n}\$ bits per bit.
 *      For a bitvector of 4GB, i.e. \f$ \log n = 35 \f$ we get about 0.01 bits per bit.
 *      If the $j-i+1 < \log^4 n$ then
 *      (2) we store the relative position of every $64$th set bit (called L2-sampled bits)
 *      in b[i..j-1] in at most $4\log\log n$ bits per L2-sampled bits.
 *      An pessimistic upper bound for the space would be
 *      \f$ \frac{4\log\log n}{64} \leq \frac{24}{64} = 0.375\f$ bit per
 *      bit (since $\log\log n\leq 6$. It is very pessimistic, since we store
 *      the relative position in $\log\log(j-i+1)\leq \log\log n$ bits.
 *
 * \tparam t_b       Bit pattern `0`,`1`,`10`,`01` which should be ranked.
 * \tparam t_pat_len Length of the bit pattern.
 *
 * The implementation is a practical variant of the following reference:
 *
 * \par Reference
 *      David Clark:
 *      PhD Thesis: Compact Pat Trees
 *      University of Waterloo, 1996 (Section 2.2.2).
 *      http://www.nlc-bnc.ca/obj/s4/f2/dsk3/ftp04/nq21335.pdf
 *
 * @ingroup select_support_group
 */
template <uint8_t t_b = 1, uint8_t t_pat_len = 1>
class select_support_mcl : public select_support
{
private:
    static_assert(t_b == 1u or t_b == 0u or t_b == 10u or t_b == 11u,
                  "select_support_mcl: bit pattern must be `0`,`1`,`10`, `01`, or `11`");
    static_assert(t_pat_len == 1u or t_pat_len == 2u, "select_support_mcl: bit pattern length must be 1 or 2");

public:
    typedef bit_vector bit_vector_type;
    enum
    {
        bit_pat = t_b
    };
    enum
    {
        bit_pat_len = t_pat_len
    };

private:
    uint32_t m_logn = 0, // \f$ log(size) \f$
        m_logn2 = 0,     // \f$ log^2(size) \f$
        m_logn4 = 0;     // \f$ log^4(size) \f$
    // entry i of m_superblock equals the answer to select_1(B,i*4096)
    int_vector<0> m_superblock;
    int_vector<0> * m_longsuperblock = nullptr;
    int_vector<0> * m_miniblock = nullptr;
    size_type m_arg_cnt = 0;
    void initData();
    void init_fast(bit_vector const * v = nullptr);

public:
    explicit select_support_mcl(bit_vector const * v = nullptr);
    select_support_mcl(select_support_mcl<t_b, t_pat_len> const & ss);
    select_support_mcl(select_support_mcl<t_b, t_pat_len> && ss);
    ~select_support_mcl();

    void init_slow(bit_vector const * v = nullptr);
    //! Select function
    inline size_type select(size_type i) const;
    //! Alias for select(i).
    inline size_type operator()(size_type i) const;
    size_type serialize(std::ostream & out, structure_tree_node * v = nullptr, std::string name = "") const;
    void load(std::istream & in, bit_vector const * v = nullptr);
    void set_vector(bit_vector const * v = nullptr);
    //!\brief Serialise (save) via cereal
    template <typename archive_t>
    void CEREAL_SAVE_FUNCTION_NAME(archive_t & ar) const;
    //!\brief Serialise (load) via cereal
    template <typename archive_t>
    void CEREAL_LOAD_FUNCTION_NAME(archive_t & ar);
    select_support_mcl<t_b, t_pat_len> & operator=(select_support_mcl const & ss);
    select_support_mcl<t_b, t_pat_len> & operator=(select_support_mcl &&);
    bool operator==(select_support_mcl const & other) const noexcept;
    bool operator!=(select_support_mcl const & other) const noexcept;
};

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len>::select_support_mcl(bit_vector const * f_v) : select_support(f_v)
{
    if (t_pat_len > 1 or (vv != nullptr and vv->size() < 100000))
        init_slow(vv);
    else
        init_fast(vv);
    return;
}

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len>::select_support_mcl(select_support_mcl const & ss) :
    select_support(ss.m_v),
    m_logn(ss.m_logn),
    m_logn2(ss.m_logn2),
    m_logn4(ss.m_logn4),
    m_superblock(ss.m_superblock),
    m_arg_cnt(ss.m_arg_cnt)
{
    size_type sb = (m_arg_cnt + 4095) >> 12;
    if (ss.m_longsuperblock != nullptr)
    {
        m_longsuperblock = new int_vector<0>[sb]; // copy longsuperblocks
        for (size_type i = 0; i < sb; ++i)
        {
            m_longsuperblock[i] = ss.m_longsuperblock[i];
        }
    }
    m_miniblock = nullptr;
    if (ss.m_miniblock != nullptr)
    {
        m_miniblock = new int_vector<0>[sb]; // copy miniblocks
        for (size_type i = 0; i < sb; ++i)
        {
            m_miniblock[i] = ss.m_miniblock[i];
        }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len>::select_support_mcl(select_support_mcl && ss) : select_support(ss.m_v)
{
    *this = std::move(ss);
}

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len> & select_support_mcl<t_b, t_pat_len>::operator=(select_support_mcl const & ss)
{
    if (this != &ss)
    {
        select_support_mcl tmp(ss);
        *this = std::move(tmp);
    }
    return *this;
}

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len> & select_support_mcl<t_b, t_pat_len>::operator=(select_support_mcl && ss)
{
    if (this != &ss)
    {
        m_logn = ss.m_logn;                        // copy log n
        m_logn2 = ss.m_logn2;                      // copy (logn)^2
        m_logn4 = ss.m_logn4;                      // copy (logn)^4
        m_superblock = std::move(ss.m_superblock); // move long superblock
        m_arg_cnt = ss.m_arg_cnt;                  // copy count of 1-bits
        m_v = ss.m_v;                              // copy pointer to the supported bit vector

        delete[] m_longsuperblock;
        m_longsuperblock = ss.m_longsuperblock;
        ss.m_longsuperblock = nullptr;

        delete[] m_miniblock;
        m_miniblock = ss.m_miniblock;
        ss.m_miniblock = nullptr;
    }
    return *this;
}

template <uint8_t t_b, uint8_t t_pat_len>
select_support_mcl<t_b, t_pat_len>::~select_support_mcl()
{
    delete[] m_longsuperblock;
    delete[] m_miniblock;
}

template <uint8_t t_b, uint8_t t_pat_len>
void select_support_mcl<t_b, t_pat_len>::init_slow(bit_vector const * v)
{
    set_vector(v);
    initData();
    if (m_v == nullptr)
        return;
    // Count the number of arguments in the bit vector
    m_arg_cnt = select_support_trait<t_b, t_pat_len>::arg_cnt(*v);

    const size_type SUPER_BLOCK_SIZE = 4096;

    if (m_arg_cnt == 0) // if there are no arguments in the vector we are done...
        return;

    size_type sb = (m_arg_cnt + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE; // number of superblocks
    delete[] m_miniblock;
    m_miniblock = new int_vector<0>[sb];

    m_superblock = int_vector<0>(sb, 0, m_logn);

    size_type arg_position[SUPER_BLOCK_SIZE], arg_cnt = 0;
    size_type sb_cnt = 0;
    for (size_type i = 0; i < v->size(); ++i)
    {
        if (select_support_trait<t_b, t_pat_len>::found_arg(i, *v))
        {
            arg_position[arg_cnt % SUPER_BLOCK_SIZE] = i;
            assert(arg_position[arg_cnt % SUPER_BLOCK_SIZE] == i);
            ++arg_cnt;
            if (arg_cnt % SUPER_BLOCK_SIZE == 0 or arg_cnt == m_arg_cnt)
            { //
                assert(sb_cnt < sb);
                m_superblock[sb_cnt] = arg_position[0];

                size_type pos_diff = arg_position[(arg_cnt - 1) % SUPER_BLOCK_SIZE] - arg_position[0];
                if (pos_diff > m_logn4)
                { // longblock
                    if (m_longsuperblock == nullptr)
                        m_longsuperblock = new int_vector<0>[sb]; // create longsuperblock
                    m_longsuperblock[sb_cnt] =
                        int_vector<0>(SUPER_BLOCK_SIZE,
                                      0,
                                      bits::hi(arg_position[(arg_cnt - 1) % SUPER_BLOCK_SIZE]) + 1);

                    for (size_type j = 0; j <= (arg_cnt - 1) % SUPER_BLOCK_SIZE; ++j)
                        m_longsuperblock[sb_cnt][j] = arg_position[j]; // copy argument positions to longsuperblock
                }
                else
                { // short block
                    m_miniblock[sb_cnt] = int_vector<0>(64, 0, bits::hi(pos_diff) + 1);
                    for (size_type j = 0; j <= (arg_cnt - 1) % SUPER_BLOCK_SIZE; j += 64)
                    {
                        m_miniblock[sb_cnt][j / 64] = arg_position[j] - arg_position[0];
                    }
                }
                ++sb_cnt;
            }
        }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
void select_support_mcl<t_b, t_pat_len>::init_fast(bit_vector const * v)
{
    set_vector(v);
    initData();
    if (m_v == nullptr)
        return;
    // Count the number of arguments in the bit vector
    m_arg_cnt = select_support_trait<t_b, t_pat_len>::arg_cnt(*v);

    const size_type SUPER_BLOCK_SIZE = 64 * 64;

    if (m_arg_cnt == 0) // if there are no arguments in the vector we are done...
        return;

    //    size_type sb = (m_arg_cnt+63+SUPER_BLOCK_SIZE-1)/SUPER_BLOCK_SIZE; // number of superblocks, add 63 as the
    //    last block could contain 63 uninitialized bits
    size_type sb = (m_arg_cnt + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE; // number of superblocks
    delete[] m_miniblock;
    m_miniblock = new int_vector<0>[sb];

    m_superblock = int_vector<0>(sb, 0, m_logn); // TODO: hier koennte man logn noch optimieren...s

    bit_vector::size_type arg_position[SUPER_BLOCK_SIZE];
    uint64_t const * data = v->data();
    uint64_t carry_new = 0;
    size_type last_k64 = 1, sb_cnt = 0;
    for (size_type i = 0, cnt_old = 0, cnt_new = 0, last_k64_sum = 1; i < (((v->bit_size() + 63) >> 6) << 6);
         i += 64, ++data)
    {
        cnt_new += select_support_trait<t_b, t_pat_len>::args_in_the_word(*data, carry_new);
        cnt_new = std::min(cnt_new,
                           m_arg_cnt); // For (0, 1), we may find nonexistent args in the padding after the bitvector.
        if (cnt_new >= last_k64_sum)
        {
            arg_position[last_k64 - 1] =
                i
                + select_support_trait<t_b, t_pat_len>::ith_arg_pos_in_the_word(*data,
                                                                                last_k64_sum - cnt_old,
                                                                                carry_new);
            last_k64 += 64;
            last_k64_sum += 64;

            if (last_k64 == SUPER_BLOCK_SIZE + 1)
            {
                m_superblock[sb_cnt] = arg_position[0];
                size_type pos_of_last_arg_in_the_block = arg_position[last_k64 - 65];

                for (size_type ii = arg_position[last_k64 - 65] + 1, j = last_k64 - 65;
                     ii < v->size() and j < SUPER_BLOCK_SIZE;
                     ++ii)
                    if (select_support_trait<t_b, t_pat_len>::found_arg(ii, *v))
                    {
                        pos_of_last_arg_in_the_block = ii;
                        ++j;
                    }
                size_type pos_diff = pos_of_last_arg_in_the_block - arg_position[0];
                if (pos_diff > m_logn4)
                { // long block
                    if (m_longsuperblock == nullptr)
                        m_longsuperblock = new int_vector<0>[sb + 1]; // create longsuperblock
                    // GEANDERT am 2010-07-17 +1 nach pos_of_last_arg..
                    m_longsuperblock[sb_cnt] =
                        int_vector<0>(SUPER_BLOCK_SIZE, 0, bits::hi(pos_of_last_arg_in_the_block) + 1);
                    for (size_type j = arg_position[0], k = 0;
                         k < SUPER_BLOCK_SIZE and j <= pos_of_last_arg_in_the_block;
                         ++j)
                        if (select_support_trait<t_b, t_pat_len>::found_arg(j, *v))
                        {
                            if (k >= SUPER_BLOCK_SIZE)
                            {
                                for (size_type ii = 0; ii < SUPER_BLOCK_SIZE; ++ii)
                                {
                                    std::cout << "(" << ii << "," << m_longsuperblock[sb_cnt][ii] << ") ";
                                }
                                std::cout << std::endl;
                                std::cout << "k=" << k << " SUPER_BLOCK_SIZE=" << SUPER_BLOCK_SIZE << std::endl;
                                std::cout << "pos_of_last_arg_in_the_block" << pos_of_last_arg_in_the_block
                                          << std::endl;
                                std::cout.flush();
                            }
                            m_longsuperblock[sb_cnt][k++] = j;
                        }
                }
                else
                {
                    m_miniblock[sb_cnt] = int_vector<0>(64, 0, bits::hi(pos_diff) + 1);
                    for (size_type j = 0; j < SUPER_BLOCK_SIZE; j += 64)
                    {
                        m_miniblock[sb_cnt][j / 64] = arg_position[j] - arg_position[0];
                    }
                }
                ++sb_cnt;
                last_k64 = 1;
            }
        }
        cnt_old = cnt_new;
    }
    // handle last block: append long superblock
    if (last_k64 > 1)
    {
        if (m_longsuperblock == nullptr)
            m_longsuperblock = new int_vector<0>[sb + 1]; // create longsuperblock
        m_longsuperblock[sb_cnt] = int_vector<0>(SUPER_BLOCK_SIZE, 0, bits::hi(v->size() - 1) + 1);
        for (size_type i = arg_position[0], k = 0; i < v->size(); ++i)
        {
            if (select_support_trait<t_b, t_pat_len>::found_arg(i, *v))
            {
                m_longsuperblock[sb_cnt][k++] = i;
            }
        }
        ++sb_cnt;
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
inline auto select_support_mcl<t_b, t_pat_len>::select(size_type i) const -> size_type
{
    assert(i > 0 and i <= m_arg_cnt);

    i = i - 1;
    size_type sb_idx = i >> 12;   // i/4096
    size_type offset = i & 0xFFF; // i%4096
    if (m_longsuperblock != nullptr and !m_longsuperblock[sb_idx].empty())
    {
        return m_longsuperblock[sb_idx][offset];
    }
    else
    {
        if ((offset & 0x3F) == 0)
        {
            assert(sb_idx < m_superblock.size());
            assert((offset >> 6) < m_miniblock[sb_idx].size());
            return m_superblock[sb_idx] + m_miniblock[sb_idx][offset >> 6 /*/64*/];
        }
        else
        {
            i = i - (sb_idx << 12) - ((offset >> 6) << 6);
            // now i > 0 and i <= 64
            assert(i > 0);
            size_type pos = m_superblock[sb_idx] + m_miniblock[sb_idx][offset >> 6] + 1;

            // now pos is the position from where we search for the ith argument
            size_type word_pos = pos >> 6;
            size_type word_off = pos & 0x3F;
            uint64_t const * data = m_v->data() + word_pos;
            uint64_t carry = select_support_trait<t_b, t_pat_len>::init_carry(data, word_pos);
            size_type args = select_support_trait<t_b, t_pat_len>::args_in_the_first_word(*data, word_off, carry);

            if (args >= i)
            {
                return (word_pos << 6)
                     + select_support_trait<t_b, t_pat_len>::ith_arg_pos_in_the_first_word(*data, i, word_off, carry);
            }
            word_pos += 1;
            size_type sum_args = args;
            carry = select_support_trait<t_b, t_pat_len>::get_carry(*data);
            uint64_t old_carry = carry;
            args = select_support_trait<t_b, t_pat_len>::args_in_the_word(*(++data), carry);
            while (sum_args + args < i)
            {
                sum_args += args;
                assert(data + 1 < m_v->data() + ((m_v->bit_size() + 63) >> 6));
                old_carry = carry;
                args = select_support_trait<t_b, t_pat_len>::args_in_the_word(*(++data), carry);
                word_pos += 1;
            }
            return (word_pos << 6)
                 + select_support_trait<t_b, t_pat_len>::ith_arg_pos_in_the_word(*data, i - sum_args, old_carry);
        }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
inline auto select_support_mcl<t_b, t_pat_len>::operator()(size_type i) const -> size_type
{
    return select(i);
}

template <uint8_t t_b, uint8_t t_pat_len>
void select_support_mcl<t_b, t_pat_len>::initData()
{
    m_arg_cnt = 0;
    if (nullptr == m_v)
    {
        m_logn = m_logn2 = m_logn4 = 0;
    }
    else
    {
        m_logn = bits::hi(((m_v->bit_size() + 63) >> 6) << 6) + 1; // TODO maybe it's better here to take a _MAX(...,12)
        m_logn2 = m_logn * m_logn;
        m_logn4 = m_logn2 * m_logn2;
    }
    delete[] m_longsuperblock;
    m_longsuperblock = nullptr;
    delete[] m_miniblock;
    m_miniblock = nullptr;
}

template <uint8_t t_b, uint8_t t_pat_len>
void select_support_mcl<t_b, t_pat_len>::set_vector(bit_vector const * v)
{
    m_v = v;
}

template <uint8_t t_b, uint8_t t_pat_len>
auto select_support_mcl<t_b, t_pat_len>::serialize(std::ostream & out, structure_tree_node * v, std::string name) const
    -> size_type
{
    structure_tree_node * child = structure_tree::add_child(v, name, util::class_name(*this));
    size_type written_bytes = 0;
    // write the number of 1-bits in the supported bit_vector
    out.write((char *)&m_arg_cnt, sizeof(size_type) / sizeof(char));
    written_bytes = sizeof(size_type) / sizeof(char);
    // number of superblocks in the data structure
    size_type sb = (m_arg_cnt + 4095) >> 12;

    if (m_arg_cnt)
    {                                                                      // if there exists 1-bits to be supported
        written_bytes += m_superblock.serialize(out, child, "superblock"); // serialize superblocks
        bit_vector mini_or_long;                                           // Helper vector: mini or long block?
        if (m_longsuperblock != nullptr)
        {
            mini_or_long.resize(sb); // resize indicator bit_vector to the number of superblocks
            for (size_type i = 0; i < sb; ++i)
                mini_or_long[i] = !m_miniblock[i].empty();
        }
        written_bytes += mini_or_long.serialize(out, child, "mini_or_long");
        size_type written_bytes_long = 0;
        size_type written_bytes_mini = 0;
        for (size_type i = 0; i < sb; ++i)
            if (!mini_or_long.empty() and !mini_or_long[i])
            {
                written_bytes_long += m_longsuperblock[i].serialize(out);
            }
            else
            {
                written_bytes_mini += m_miniblock[i].serialize(out);
            }
        written_bytes += written_bytes_long;
        written_bytes += written_bytes_mini;
        structure_tree_node * child_long =
            structure_tree::add_child(child, "longsuperblock", util::class_name(m_longsuperblock));
        structure_tree::add_size(child_long, written_bytes_long);
        structure_tree_node * child_mini =
            structure_tree::add_child(child, "minisuperblock", util::class_name(m_miniblock));
        structure_tree::add_size(child_mini, written_bytes_mini);
    }
    structure_tree::add_size(child, written_bytes);
    return written_bytes;
}

template <uint8_t t_b, uint8_t t_pat_len>
void select_support_mcl<t_b, t_pat_len>::load(std::istream & in, bit_vector const * v)
{
    set_vector(v);
    initData();
    // read the number of 1-bits in the supported bit_vector
    in.read((char *)&m_arg_cnt, sizeof(size_type) / sizeof(char));
    size_type sb = (m_arg_cnt + 4095) >> 12;

    if (m_arg_cnt)
    {                          // if there exists 1-bits to be supported
        m_superblock.load(in); // load superblocks

        delete[] m_miniblock;
        m_miniblock = nullptr;
        delete[] m_longsuperblock;
        m_longsuperblock = nullptr;

        bit_vector mini_or_long;             // Helper vector: mini or long block?
        mini_or_long.load(in);               // Load the helper vector
        m_miniblock = new int_vector<0>[sb]; // Create miniblock int_vector<0>
        if (!mini_or_long.empty())
            m_longsuperblock = new int_vector<0>[sb]; // Create longsuperblock int_vector<0>

        for (size_type i = 0; i < sb; ++i)
            if (!mini_or_long.empty() and not mini_or_long[i])
            {
                m_longsuperblock[i].load(in);
            }
            else
            {
                m_miniblock[i].load(in);
            }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
template <typename archive_t>
void select_support_mcl<t_b, t_pat_len>::CEREAL_SAVE_FUNCTION_NAME(archive_t & ar) const
{
    ar(CEREAL_NVP(m_arg_cnt));
    ar(CEREAL_NVP(m_logn));
    ar(CEREAL_NVP(m_logn2));
    ar(CEREAL_NVP(m_logn4));
    size_type sb = (m_arg_cnt + 4095) >> 12;
    if (m_arg_cnt)
    {
        ar(CEREAL_NVP(m_superblock));
        bit_vector mini_or_long;
        if (m_longsuperblock != nullptr)
        {
            mini_or_long.resize(sb);
            for (size_type i = 0; i < sb; ++i)
            {
                mini_or_long[i] = !m_miniblock[i].empty();
            }
        }
        ar(CEREAL_NVP(mini_or_long));
        for (size_type i = 0; i < sb; ++i)
        {
            if (!mini_or_long.empty() and !mini_or_long[i])
            {
                ar(CEREAL_NVP(m_longsuperblock[i]));
            }
            else
            {
                ar(CEREAL_NVP(m_miniblock[i]));
            }
        }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
template <typename archive_t>
void select_support_mcl<t_b, t_pat_len>::CEREAL_LOAD_FUNCTION_NAME(archive_t & ar)
{
    delete[] m_longsuperblock;
    m_longsuperblock = nullptr;
    delete[] m_miniblock;
    m_miniblock = nullptr;

    ar(CEREAL_NVP(m_arg_cnt));
    ar(CEREAL_NVP(m_logn));
    ar(CEREAL_NVP(m_logn2));
    ar(CEREAL_NVP(m_logn4));

    size_type sb = (m_arg_cnt + 4095) >> 12;

    if (m_arg_cnt)
    {
        ar(CEREAL_NVP(m_superblock));

        delete[] m_miniblock;
        m_miniblock = nullptr;
        delete[] m_longsuperblock;
        m_longsuperblock = nullptr;

        bit_vector mini_or_long;
        ar(CEREAL_NVP(mini_or_long));
        m_miniblock = new int_vector<0>[sb];

        if (!mini_or_long.empty())
        {
            m_longsuperblock = new int_vector<0>[sb];
        }

        for (size_type i = 0; i < sb; ++i)
        {
            if (!mini_or_long.empty() and !mini_or_long[i])
            {
                ar(CEREAL_NVP(m_longsuperblock[i]));
            }
            else
            {
                ar(CEREAL_NVP(m_miniblock[i]));
            }
        }
    }
}

template <uint8_t t_b, uint8_t t_pat_len>
bool select_support_mcl<t_b, t_pat_len>::operator==(select_support_mcl const & other) const noexcept
{
    return (m_logn == other.m_logn) && (m_logn2 == other.m_logn2) && (m_logn4 == other.m_logn4)
        && (m_superblock == other.m_superblock) && (m_arg_cnt == other.m_arg_cnt)
        && ((m_longsuperblock == nullptr && other.m_longsuperblock == nullptr)
            || (*m_longsuperblock == *other.m_longsuperblock))
        && ((m_miniblock == other.m_miniblock) || (*m_miniblock == *other.m_miniblock));
}

template <uint8_t t_b, uint8_t t_pat_len>
bool select_support_mcl<t_b, t_pat_len>::operator!=(select_support_mcl const & other) const noexcept
{
    return !(*this == other);
}
} // namespace sdsl

#endif
