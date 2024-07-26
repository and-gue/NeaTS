#include "sdsl/sd_vector.hpp"
#include "sdsl/simple_sds.hpp"

#include <cassert>

//! Namespace for the succinct data structure library
namespace sdsl
{

//-----------------------------------------------------------------------------

sd_vector_builder::sd_vector_builder() :
    m_size(0), m_capacity(0),
    m_wl(0),
    m_tail(0), m_tail_inc(0), m_items(0),
    m_last_high(0), m_highpos(0)
{
}

sd_vector_builder::sd_vector_builder(size_type n, size_type m, bool multiset) :
    m_size(n), m_capacity(m),
    m_wl(0),
    m_tail(0), m_tail_inc((multiset ? 0 : 1)), m_items(0),
    m_last_high(0), m_highpos(0)
{
    if (!multiset && m_capacity > m_size) {
        throw std::runtime_error("sd_vector_builder: requested capacity is larger than vector size");
    }

    std::pair<size_type, size_type> params = sd_vector<>::get_params(m_size, m_capacity);
    m_wl = params.first;
    m_low = int_vector<>(m_capacity, 0, params.first);
    m_high = bit_vector(params.second, 0);
}

void
sd_vector_builder::swap(sd_vector_builder& sdb)
{
    std::swap(m_size, sdb.m_size);
    std::swap(m_capacity, sdb.m_capacity);
    std::swap(m_wl, sdb.m_wl);
    std::swap(m_tail, sdb.m_tail);
    std::swap(m_tail_inc, sdb.m_tail_inc);
    std::swap(m_items, sdb.m_items);
    std::swap(m_last_high, sdb.m_last_high);
    std::swap(m_highpos, sdb.m_highpos);
    m_low.swap(sdb.m_low);
    m_high.swap(sdb.m_high);
}

//-----------------------------------------------------------------------------

template<>
sd_vector<>::sd_vector(builder_type& builder)
{
    if(builder.items() != builder.capacity()) {
      throw std::runtime_error("sd_vector: the builder is not at full capacity.");
    }

    m_size = builder.m_size;
    m_wl = builder.m_wl;
    m_low.swap(builder.m_low);
    m_high.swap(builder.m_high);
    util::init_support(m_high_1_select, &m_high);
    util::init_support(m_high_0_select, &m_high);

    builder = builder_type();
}

template<>
void
sd_vector<>::simple_sds_serialize(std::ostream& out) const
{
    simple_sds::serialize_value<size_t>(this->m_size, out);

    // The vector may have been built with another SDSL fork or with an older version
    // of this fork. If the number of buckets is too high, we serialize a copy of the
    // `high` bitvector instead.
    size_type buckets = get_buckets(this->size(), this->m_low.width());
    if (this->m_high.size() > buckets + this->ones()) {
        bit_vector high_copy = this->m_high;
        high_copy.resize(buckets + this->ones());
        high_copy.simple_sds_serialize(out);
    }
    else {
        this->m_high.simple_sds_serialize(out);
    }

    this->m_low.simple_sds_serialize(out);
}

template<>
void
sd_vector<>::simple_sds_load(std::istream& in)
{
    size_t length = simple_sds::load_value<size_t>(in);
    hi_bit_vector_type high; high.simple_sds_load(in);
    int_vector<> low; low.simple_sds_load(in);

    // It may be that `low.size() > length` because we have a very dense multiset.
    if (high.size() != low.size() + get_buckets(length, low.width())) {
        throw simple_sds::InvalidData("Invalid number of buckets");
    }

    this->m_size = length;
    this->m_wl = low.width();
    this->m_high = std::move(high);
    util::init_support(this->m_high_1_select, &(this->m_high));
    util::init_support(this->m_high_0_select, &(this->m_high));
    this->m_low = std::move(low);
}

template<>
size_t
sd_vector<>::simple_sds_size() const
{
    // Same correction for old vectors as in `simple_sds_serialize()`.
    size_t buckets = get_buckets(this->size(), this->m_low.width());
    return simple_sds::value_size<size_t>() + bit_vector::simple_sds_size(buckets + this->ones()) + this->m_low.simple_sds_size();
}

template<>
size_t
sd_vector<>::simple_sds_size(size_t n, size_t m)
{
    std::pair<size_type, size_type> params = get_params(n, m);
    return simple_sds::value_size<size_t>() + bit_vector::simple_sds_size(params.second) + int_vector<0>::simple_sds_size(m, params.first);
}

//-----------------------------------------------------------------------------

} // end namespace
