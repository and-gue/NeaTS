#include "sdsl/rle_vector.hpp"
#include "sdsl/bit_vectors.hpp"
#include "gtest/gtest.h"

using namespace sdsl;
using namespace std;

namespace
{

const size_t BV_SIZE = 1000000;

std::vector<std::pair<uint64_t, uint64_t>>
generate_runs(size_t size, size_t inverse_flip_probability)
{
    std::mt19937_64 rng;
    std::uniform_int_distribution<uint64_t> distribution(0, inverse_flip_probability);
    auto dice = bind(distribution, rng);

    std::pair<uint64_t, uint64_t> run(0, 0);
    bool ones = false;
    std::vector<std::pair<uint64_t, uint64_t>> result;
    for (size_t i = 0; i < size; ++i) {
        if (ones) {
            if (dice() == 0) {
                result.push_back(run);
                ones = false;
            } else {
                run.second++;
            }
        } else {
            if (dice() == 0) {
                run.first = i; run.second = 1;
                ones = true;
            }
        }
    }
    if (ones) {
        result.push_back(run);
    }

    return result;
}

size_t count_ones(const std::vector<std::pair<uint64_t, uint64_t>>& runs)
{
    size_t result = 0;
    for (auto run : runs) { result += run.second; }
    return result;
}

bit_vector
mark_runs(const std::vector<std::pair<uint64_t, uint64_t>>& runs, size_t size)
{
    bit_vector result(size);
    for (auto run : runs) {
        for (uint64_t i = run.first; i < run.first + run.second; i++) {
            result[i] = 1;
        }
    }
    return result;
}

template<class T>
class rle_vector_test : public ::testing::Test { };

using testing::Types;

typedef Types< rle_vector<64>, rle_vector<128> > Implementations;

TYPED_TEST_CASE(rle_vector_test, Implementations);

TYPED_TEST(rle_vector_test, from_bit_vector)
{
    std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE, 9);
    bit_vector bv = mark_runs(runs, BV_SIZE);
    size_t ones = count_ones(runs);
    TypeParam rlv = bv;

    ASSERT_EQ(rlv.size(), BV_SIZE);
    ASSERT_EQ(rlv.ones(), ones);
    for (size_t i = 0; i < bv.size(); i++) {
        ASSERT_EQ((bool)rlv[i], (bool)bv[i]);
    }
}

TYPED_TEST(rle_vector_test, from_builder)
{
    std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE, 9);
    bit_vector bv = mark_runs(runs, BV_SIZE);
    size_t ones = count_ones(runs);

    typename TypeParam::builder_type builder(BV_SIZE);
    for (auto run : runs) {
        builder.set(run.first, run.second);
    }
    TypeParam rlv(builder);

    ASSERT_EQ(rlv.size(), BV_SIZE);
    ASSERT_EQ(rlv.ones(), ones);
    ASSERT_EQ(rlv.runs(), runs.size());
    for (size_t i = 0; i < bv.size(); i++) {
        ASSERT_EQ((bool)rlv[i], (bool)bv[i]);
    }
}

TYPED_TEST(rle_vector_test, equality)
{
    TypeParam original;
    {
        std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE, 9);
        typename TypeParam::builder_type builder(BV_SIZE);
        for (auto run : runs) {
            builder.set(run.first, run.second);
        }
        original = TypeParam(builder);
    }

    TypeParam copy(original);
    ASSERT_EQ(copy, original);

    TypeParam longer_runs;
    {
        std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE, 12);
        typename TypeParam::builder_type builder(BV_SIZE);
        for (auto run : runs) {
            builder.set(run.first, run.second);
        }
        longer_runs = TypeParam(builder);
    }
    ASSERT_NE(longer_runs, original);

    TypeParam shorter;
    {
        std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE / 2, 9);
        typename TypeParam::builder_type builder(BV_SIZE / 2);
        for (auto run : runs) {
            builder.set(run.first, run.second);
        }
        shorter = TypeParam(builder);
    }
    ASSERT_NE(shorter, original);
}

TYPED_TEST(rle_vector_test, special_cases)
{
    std::vector<std::pair<uint64_t, uint64_t>> runs = generate_runs(BV_SIZE, 9);
    typename TypeParam::builder_type builder(BV_SIZE);
    for (auto run : runs) {
        builder.set(run.first, run.second);
    }
    TypeParam rlv(builder);

    typename TypeParam::rank_1_type rs_1(&rlv);
    ASSERT_EQ(rs_1(rlv.size()), rlv.ones());

    typename TypeParam::rank_0_type rs_0(&rlv);
    ASSERT_EQ(rs_0(rlv.size()), rlv.size() - rlv.ones());

    typename TypeParam::select_1_type ss(&rlv);
    ASSERT_EQ(ss(0), static_cast<typename TypeParam::size_type>(-1));
    ASSERT_EQ(ss(rlv.ones() + 1), rlv.size());
}

TYPED_TEST(rle_vector_test, builder_exceptions)
{
    {
        // Position is too small.
        typename TypeParam::builder_type builder(1024);
        builder.set(128);
        ASSERT_THROW(builder.set(128), std::runtime_error);
    }
    {
        // Position is too large.
        typename TypeParam::builder_type builder(1024);
        ASSERT_THROW(builder.set(1024), std::runtime_error);
    }
}

} // end namespace

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

