#include "sdsl/simple_sds.hpp"

#include <algorithm>

//! Namespace for SDSL.
namespace sdsl
{

//! Namespace for the Simple-SDS serialization format.
namespace simple_sds
{

//-----------------------------------------------------------------------------

// Vectors are serialized in blocks of 32 MiB.
constexpr size_t BLOCK_SIZE = 32 * 1048576;

//-----------------------------------------------------------------------------

// Returns the number of bytes of padding required after `bytes` bytes of data.
size_t padding_length(size_t bytes)
{
    size_t overflow = bytes & (sizeof(element_type) - 1);
    return (overflow > 0 ? sizeof(element_type) - overflow : 0);
}

void serialize_data(const char* buffer, size_t size, std::ostream& out)
{
    for (size_t i = 0; i < size; i += BLOCK_SIZE) {
        size_t length = std::min(BLOCK_SIZE, size - i);
        out.write(buffer + i, length);
    }

    size_t padding = padding_length(size);
    if (padding > 0) {
        size_t zero = 0;
        out.write(reinterpret_cast<const char*>(&zero), padding);
    }
}

void load_data(char* buffer, size_t size, std::istream& in)
{
    for (size_t i = 0; i < size; i += BLOCK_SIZE) {
        size_t length = std::min(BLOCK_SIZE, size - i);
        in.read(buffer + i, length);
    }

    size_t padding = padding_length(size);
    if (padding > 0) {
        in.ignore(padding);
    }
}

size_t data_size(size_t bytes) {
    return (bytes + padding_length(bytes)) / sizeof(element_type);
}

//-----------------------------------------------------------------------------

void serialize_string(const std::string& value, std::ostream& out)
{
    serialize_value(value.length(), out);
    serialize_data(value.data(), value.length(), out);
}

std::string load_string(std::istream& in)
{
    std::string result(load_value<size_t>(in), '\0');
    if (result.length() > 0) {
        load_data(&(result.front()), result.length(), in);
    }
    return result;
}

size_t string_size(const std::string& value)
{
    return value_size(value.size()) + data_size(value.length());
}

size_t string_size(size_t n)
{
    return value_size<size_t>() + data_size(n);
}

//-----------------------------------------------------------------------------

void empty_option(std::ostream& out)
{
    size_t size = 0;
    serialize_value(size, out);
}

void skip_option(std::istream& in)
{
    size_t size = load_value<size_t>(in);
    // Seeking could be faster, but the stream might not be seekable.
    in.ignore(size * sizeof(element_type));
}

size_t empty_option_size()
{
    return value_size<size_t>();
}

//-----------------------------------------------------------------------------

} // namespace simple_sds

} // namespace sdsl
