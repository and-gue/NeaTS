#include "sdsl/simple_sds.hpp"
#include "sdsl/int_vector.hpp"
#include "sdsl/sd_vector.hpp"
#include "gtest/gtest.h"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace sdsl;

namespace
{

//-----------------------------------------------------------------------------

struct ByteArray
{
    std::vector<std::uint8_t> bytes;
    size_t                    sum = 0;

    void simple_sds_serialize(std::ostream& out) const
    {
        simple_sds::serialize_vector(this->bytes, out);
        simple_sds::serialize_value(this->sum, out);
    }

    void simple_sds_load(std::istream& in)
    {
        this->bytes = simple_sds::load_vector<std::uint8_t>(in);
        this->sum = simple_sds::load_value<size_t>(in);
        size_t real_sum = 0;
        for (auto value : this->bytes) { real_sum += value; }
        if (real_sum != this->sum) {
            throw simple_sds::InvalidData("Incorrect sum");
        }
    }

    size_t simple_sds_size() const
    {
        return simple_sds::vector_size(this->bytes) + simple_sds::value_size(this->sum);
    }

    bool operator==(const ByteArray& another) const
    {
        return (this->bytes == another.bytes && this->sum == another.sum);
    }
};

struct ComplexStructure
{
    std::pair<size_t, size_t> header;
    bool                      has_byte_array;
    ByteArray                 byte_array;
    std::vector<double>       numbers;

    void simple_sds_serialize(std::ostream& out) const
    {
        simple_sds::serialize_value(this->header, out);
        if (this->has_byte_array) {
            simple_sds::serialize_option(this->byte_array, out);
        } else {
            simple_sds::empty_option(out);
        }
        simple_sds::serialize_vector(this->numbers, out);
    }

    void simple_sds_load(std::istream& in)
    {
        this->header = simple_sds::load_value<std::pair<size_t, size_t>>(in);
        this->has_byte_array = simple_sds::load_option(this->byte_array, in);
        this->numbers = simple_sds::load_vector<double>(in);
    }

    size_t simple_sds_size() const
    {
        size_t result = simple_sds::value_size(this->header);
        result += (this->has_byte_array ? simple_sds::option_size(this->byte_array) : simple_sds::empty_option_size());
        result += simple_sds::vector_size(this->numbers);
        return result;
    }

    bool operator==(const ComplexStructure& another) const
    {
        return (this->header == another.header &&
                this->has_byte_array == another.has_byte_array &&
                this->byte_array == another.byte_array &&
                this->numbers == another.numbers);
    }
};

//-----------------------------------------------------------------------------

std::string temp_file_name()
{
    char buffer[] = "simple-sds-XXXXXX";
    int fail = mkstemp(buffer);
    if (fail == -1) { return std::string(); }
    else { return std::string(buffer); }
}

size_t file_size(const std::string& filename)
{
    struct stat stat_buf;
    if (stat(filename.c_str(), &stat_buf) == 0) { return stat_buf.st_size; }
    else { return 0; }
}

bit_vector random_bit_vector(size_t size, double density)
{
    bit_vector result(size, 0);
    std::mt19937_64 rng(0xDEADBEEF);

    if (density == 0.5) {
        for (size_t i = 0; i < result.size(); i += 64) {
            std::uint64_t value = rng() & bits::lo_set[std::min(size_t(result.size() - i), size_t(64))];
            result.set_int(i, value);
        }
    } else {
        std::bernoulli_distribution distribution(density);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = distribution(rng);
        }
    }

    return result;
}

template<uint8_t t_width>
int_vector<t_width> random_int_vector(size_t size)
{
    int_vector<t_width> result(size, 0);
    std::mt19937_64 rng(0xDEADBEEF);

    for (size_t i = 0; i < result.size(); i++) {
        result[i] = rng();
    }

    return result;
}

//-----------------------------------------------------------------------------

class BasicStructures : public ::testing::Test
{
public:
    void check(const ComplexStructure& original, size_t expected_size) const
    {
        ASSERT_EQ(original.simple_sds_size(), expected_size) << "Invalid serialization size in elements";

        std::string filename = temp_file_name();
        ASSERT_NE(filename, "") << "Temporary file creation failed";

        simple_sds::serialize_to(original, filename);
        size_t bytes = file_size(filename);
        EXPECT_EQ(bytes, expected_size * sizeof(simple_sds::element_type)) << "Invalid file size";

        ComplexStructure loaded;
        simple_sds::load_from(loaded, filename);
        EXPECT_EQ(loaded, original) << "Invalid loaded structure";

        std::ifstream in(filename, std::ios_base::binary);
        in.seekg(simple_sds::value_size(original.header) * sizeof(simple_sds::element_type));
        simple_sds::skip_option(in);
        std::vector<double> numbers = simple_sds::load_vector<double>(in);
        EXPECT_EQ(numbers, original.numbers) << "Invalid numbers after skipping the optional structure";

        std::remove(filename.c_str());
    }
};

TEST_F(BasicStructures, Empty)
{
    ComplexStructure original = {
        { 123, 456, },
        false,
        { { }, 0, },
        { },
    };
    size_t expected_size = 2 + 1 + 1;
    this->check(original, expected_size);
}

TEST_F(BasicStructures, Numbers)
{
    ComplexStructure original = {
        { 123, 456, },
        false,
        { { }, 0, },
        { 1.0, 2.0, 3.0, 5.0, },
    };
    size_t expected_size = 2 + 1 + 5;
    this->check(original, expected_size);
}

TEST_F(BasicStructures, EmptyBytes)
{
    ComplexStructure original = {
        { 123, 456, },
        true,
        { { }, 0, },
        { },
    };
    size_t expected_size = 2 + 3 + 1;
    this->check(original, expected_size);
}

TEST_F(BasicStructures, BytesWithPadding)
{
    ComplexStructure original = {
        { 123, 456, },
        true,
        { { 1, 2, 3, 5, 8, }, 19, },
        { },
    };
    size_t expected_size = 2 + 4 + 1;
    this->check(original, expected_size);
}

TEST_F(BasicStructures, BytesAndNumbers)
{
    ComplexStructure original = {
        { 123, 456, },
        true,
        { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, }, 136, },
        { 1.0, 2.0, 3.0, 5.0 },
    };
    size_t expected_size = 2 + 5 + 5;
    this->check(original, expected_size);
}

TEST_F(BasicStructures, BytesAndNumbersWithPadding)
{
    ComplexStructure original = {
        { 123, 456, },
        true,
        { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, }, 120, },
        { 1.0, 2.0, 3.0, 5.0 },
    };
    size_t expected_size = 2 + 5 + 5;
    this->check(original, expected_size);
}

//-----------------------------------------------------------------------------

class IntegerVector : public ::testing::Test
{
public:
    void check(const int_vector<0>& original, size_t expected_size) const
    {
        ASSERT_EQ(original.simple_sds_size(), expected_size) << "Invalid serialization size in elements";
        EXPECT_EQ(int_vector<0>::simple_sds_size(original.size(), original.width()), expected_size) << "Invalid size estimate";

        std::string filename = temp_file_name();
        ASSERT_NE(filename, "") << "Temporary file creation failed";

        simple_sds::serialize_to(original, filename);
        size_t bytes = file_size(filename);
        EXPECT_EQ(bytes, expected_size * sizeof(simple_sds::element_type)) << "Invalid file size";

        int_vector<0> loaded;
        simple_sds::load_from(loaded, filename);
        EXPECT_EQ(loaded, original) << "Invalid loaded structure";

        std::remove(filename.c_str());
    }
};

TEST_F(IntegerVector, Empty)
{
    int_vector<0> original;
    size_t expected_size = 4 + 0;
    this->check(original, expected_size);
}

TEST_F(IntegerVector, EmptyWithWidth)
{
    int_vector<0> original;
    original.width(13);
    size_t expected_size = 4 + 0;
    this->check(original, expected_size);
}

TEST_F(IntegerVector, WithPadding)
{
    int_vector<0> original(30, 0, 15);
    for (size_t i = 0; i < original.size(); i++) { original[i] = 1 + i * (i + 1); }
    size_t expected_size = 4 + 8;
    this->check(original, expected_size);
}

TEST_F(IntegerVector, NoPadding)
{
    int_vector<0> original(64, 0, 17);
    for (size_t i = 0; i < original.size(); i++) { original[i] = 1 + i * (i + 1); }
    size_t expected_size = 4 + 17;
    this->check(original, expected_size);
}

//-----------------------------------------------------------------------------

class BitVector : public ::testing::Test
{
public:
    void check(const bit_vector& original, size_t expected_size) const
    {
        ASSERT_EQ(original.simple_sds_size(), expected_size) << "Invalid serialization size in elements";
        EXPECT_EQ(bit_vector::simple_sds_size(original.size()), expected_size) << "Invalid size estimate";

        std::string filename = temp_file_name();
        ASSERT_NE(filename, "") << "Temporary file creation failed";

        simple_sds::serialize_to(original, filename);
        size_t bytes = file_size(filename);
        EXPECT_EQ(bytes, expected_size * sizeof(simple_sds::element_type)) << "Invalid file size";

        bit_vector loaded;
        simple_sds::load_from(loaded, filename);
        EXPECT_EQ(loaded, original) << "Invalid loaded structure";

        std::remove(filename.c_str());
    }
};

TEST_F(BitVector, Empty)
{
    bit_vector original;
    size_t expected_size = 3 + 0 + 3;
    this->check(original, expected_size);
}

TEST_F(BitVector, WithPadding)
{
    bit_vector original = random_bit_vector(515, 0.5);
    size_t expected_size = 3 + 9 + 3;
    this->check(original, expected_size);
}

TEST_F(BitVector, NoPadding)
{
    bit_vector original = random_bit_vector(448, 0.5);
    size_t expected_size = 3 + 7 + 3;
    this->check(original, expected_size);
}

//-----------------------------------------------------------------------------

class Vector : public ::testing::Test
{
public:
    template<typename Item, uint8_t t_width>
    void check(const int_vector<t_width>& original, size_t expected_size) const
    {
        ASSERT_EQ(original.simple_sds_size(), expected_size) << "Invalid serialization size in elements for t_width = " << unsigned(t_width);
        EXPECT_EQ(int_vector<t_width>::simple_sds_size(original.size()), expected_size) << "Invalid size estimate";

        std::string filename = temp_file_name();
        ASSERT_NE(filename, "") << "Temporary file creation failed for t_width = " << unsigned(t_width);

        simple_sds::serialize_to(original, filename);
        size_t bytes = file_size(filename);
        EXPECT_EQ(bytes, expected_size * sizeof(simple_sds::element_type)) << "Invalid file size for t_width = " << unsigned(t_width);

        int_vector<t_width> loaded;
        simple_sds::load_from(loaded, filename);
        EXPECT_EQ(loaded, original) << "Invalid loaded structure for t_width = " << unsigned(t_width);

        std::ifstream in(filename, std::ios_base::binary);
        std::vector<Item> as_vector = simple_sds::load_vector<Item>(in);
        in.close();

        // Remove the temporary file first in case the length test fails.
        std::remove(filename.c_str());

        ASSERT_EQ(as_vector.size(), original.size()) << "Invalid std::vector size for t_width = " << unsigned(t_width);
        bool ok = true;
        for (size_t i = 0; i < as_vector.size(); i++) {
            if (as_vector[i] != original[i]) {
                ok = false; break;
            }
        }
        EXPECT_TRUE(ok) << "Invalid std::vector values for t_width = " << unsigned(t_width);
    }
};

TEST_F(Vector, Empty)
{
    int_vector<8> bytes;
    size_t bytes_size = 1 + 0;
    this->check<std::uint8_t, 8>(bytes, bytes_size);

    int_vector<64> words;
    size_t words_size = 1 + 0;
    this->check<std::uint64_t, 64>(words, words_size);
}

TEST_F(Vector, BytesWithPadding)
{
    int_vector<8> bytes = random_int_vector<8>(123);
    size_t bytes_size = 1 + 16;
    this->check<std::uint8_t, 8>(bytes, bytes_size);
}

TEST_F(Vector, NoPadding)
{
    int_vector<8> bytes = random_int_vector<8>(96);
    size_t bytes_size = 1 + 12;
    this->check<std::uint8_t, 8>(bytes, bytes_size);

    int_vector<64> words = random_int_vector<64>(14);
    size_t words_size = 1 + 14;
    this->check<std::uint64_t, 64>(words, words_size);
}

//-----------------------------------------------------------------------------

class SparseVector : public ::testing::Test
{
public:
    void check(const sd_vector<>& original) const
    {
        size_t expected_size = original.simple_sds_size();
        EXPECT_EQ(sd_vector<>::simple_sds_size(original.size(), original.ones()), expected_size) << "Invalid size estimate";

        std::string filename = temp_file_name();
        ASSERT_NE(filename, "") << "Temporary file creation failed";

        simple_sds::serialize_to(original, filename);
        size_t bytes = file_size(filename);
        EXPECT_EQ(bytes, expected_size * sizeof(simple_sds::element_type)) << "Invalid file size";

        sd_vector<> loaded;
        simple_sds::load_from(loaded, filename);
        EXPECT_EQ(loaded, original) << "Invalid loaded structure";

        std::remove(filename.c_str());
    }
};

TEST_F(SparseVector, Empty)
{
    bit_vector empty;
    sd_vector<> original(empty);
    this->check(original);
}

TEST_F(SparseVector, Sparse)
{
    bit_vector sparse = random_bit_vector(542, 0.02);
    sd_vector<> original(sparse);
    this->check(original);
}

TEST_F(SparseVector, Dense)
{
    bit_vector dense = random_bit_vector(621, 0.5);
    sd_vector<> original(dense);
    this->check(original);
}

//-----------------------------------------------------------------------------

} // anonymous namespace

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//-----------------------------------------------------------------------------
