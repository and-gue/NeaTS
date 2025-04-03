/* sdsl - succinct data structures library
    Copyright (C) 2021 Jouni Sirén

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/ .
*/
/*! \file simple_sds.hpp
    \brief simple_sds.hpp implements the basics of the simple-sds serialization format.
	\author Jouni Sirén
*/

#ifndef INCLUDED_SDSL_SIMPLE_SDS
#define INCLUDED_SDSL_SIMPLE_SDS

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

//! Namespace for SDSL.
namespace sdsl
{

//! Namespace for the simple-sds serialization format.
/*! \par The simple-sds serialization format is defined in
 *  https://github.com/jltsiren/simple-sds/blob/main/SERIALIZATION.md.
 *  This namespace implements serializing and loading basic types:
 *  serializable types, vectors, and optional structures.
 *
 *  \par Error handling is based on exceptions. Any exceptions from the input
 *  and output streams are passed through. Implementations may use the
 *  provided `InvalidData` exception to indicate that the loaded data failed
 *  sanity checks.
 *
 *  \par The serialization format is based on elements, which are unsigned
 *  64-bit little-endian integers. Any structure that implements the
 *  serialization interface can be serialized:
 *
 *  - `void simple_sds_serialize(std::ofstream&) const`: Serialize the structure
 *  into the stream.
 *  - `void simple_sds_load(std::ifstream&)`: Load the structure
 *  from the stream.
 *  - `size_t simple_sds_size() const`: Number of elements needed for
 *  serializing the structure.
 */
namespace simple_sds
{

//-----------------------------------------------------------------------------

//! Serialization is defined in terms of unsigned 64-bit little-endian integers.
typedef std::uint64_t element_type;

//! Number of bits in an element.
constexpr size_t ELEMENT_BITS = 64;

//! Number of elements required for `bits` bits.
inline size_t bits_to_elements(size_t bits)
{
    return (bits + ELEMENT_BITS - 1) / ELEMENT_BITS;
}

//! An exception that indicates that the loaded data failed sanity checks.
class InvalidData : public std::runtime_error
{
public:
    //! Constructor from a string.
    /*! \param message Message returned by the `what()` method.
     */
    explicit InvalidData(const std::string& message) : std::runtime_error(message) {}

    //! Constructor from a C string.
    /*! \param message Message returned by the `what()` method.
     */
    explicit InvalidData(const char* message) : std::runtime_error(message) {}
};

//! An exception that indicates that the file could not be opened.
class CannotOpenFile : public std::runtime_error
{
public:
    //! Constructor from a string.
    /*! \param filename Name of the file.
     *  \param for_writing `true` if the file was to be opened for writing.
     */
    explicit CannotOpenFile(const std::string& filename, bool for_writing) :
        std::runtime_error(msg(filename, for_writing)) {
    }

    static std::string msg(const std::string& filename, bool for_writing) {
        std::string msg = "Cannot open ";
        msg += filename;
        msg += (for_writing ? " for writing" : " for reading");
        return msg;
    }
};

//! Serialize the data in a buffer.
/*! \param buffer A non-null pointer to the buffer.
 *  \param size Size of the buffer in bytes.
 *  \param out Output stream for serialization.
 *
 *  \par The buffer will be serialized in 32 MiB blocks.
 *  The data will be padded with 0-bytes to the next multiple of 8 bytes.
 */
void serialize_data(const char* buffer, size_t size, std::ostream& out);

//! Load data into a buffer.
/*! \param buffer A non-null pointer to the buffer.
 *  \param size Size of the buffer in bytes.
 *  \param out Input stream for loading the data.
 *
 *  \par The buffer will be loaded in 32 MiB blocks.
 */
void load_data(char* buffer, size_t size, std::istream& in);

//! Size of the data in elements
/*! \param size Size of the data in bytes.
 *  \return Number of elements needed for serializing the data.
 */
size_t data_size(size_t bytes);

//-----------------------------------------------------------------------------

//! Serialize a serializable value.
/*! \tparam Serializable A fixed-size type with the size a multiple of 8 bytes.
 *  \param value Value to be serialized.
 *  \param out Output stream for serialization.
 *
 *  \par This corresponds to a serializable type in simple-sds.
 */
template<typename Serializable>
void serialize_value(const Serializable& value, std::ostream& out)
{
    static_assert(sizeof(Serializable) % sizeof(element_type) == 0, "The size of a serializable type must be a multiple of 8 bytes");
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

//! Load a serializable value.
/*! \tparam Serializable A fixed-size type with the size a multiple of 8 bytes.
 *  \param int Input stream for loading the data.
 *  \return The loaded value.
 *
 *  \par This corresponds to a serializable type in simple-sds.
 */
template<typename Serializable>
Serializable load_value(std::istream& in)
{
    static_assert(sizeof(Serializable) % sizeof(element_type) == 0, "The size of a serializable type must be a multiple of 8 bytes");
    Serializable value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

//! Size of a serializable type in elements.
/*! \tparam Serializable A fixed-size type with the size a multiple of 8 bytes.
 *  \return Number of elements needed for serializing the type.
 *
 *  \par This corresponds to a serializable type in simple-sds.
 */
template<typename Serializable>
size_t value_size()
{
    static_assert(sizeof(Serializable) % sizeof(element_type) == 0, "The size of a serializable type must be a multiple of 8 bytes");
    return sizeof(Serializable) / sizeof(element_type);
}

//! Size of a serializable type in elements.
/*! \tparam Serializable A fixed-size type with the size a multiple of 8 bytes.
 *  \param value A serializable value.
 *  \return Number of elements needed for serializing the type.
 *
 *  \par This corresponds to a serializable type in simple-sds.
 */
template<typename Serializable>
size_t value_size(const Serializable& value)
{
    static_assert(sizeof(Serializable) % sizeof(element_type) == 0, "The size of a serializable type must be a multiple of 8 bytes");
    return sizeof(value) / sizeof(element_type);
}

//-----------------------------------------------------------------------------

//! Serialize a vector of items.
/*! \tparam Item A fixed-size type with the size either 1 byte or a multiple of 8 bytes.
 *  \param value The vector to be serialized.
 *  \param out Output stream for serialization.
 *
 *  \par This corresponds to a vector of serializable items or bytes in simple-sds.
 */
template<typename Item>
void serialize_vector(const std::vector<Item>& value, std::ostream& out)
{
    static_assert(sizeof(Item) == 1 || sizeof(Item) % sizeof(element_type) == 0, "The size of an item must be 1 byte or a multiple of 8 bytes");
    serialize_value(value.size(), out);
    serialize_data(reinterpret_cast<const char*>(value.data()), value.size() * sizeof(Item), out);
}

//! Load a vector of items.
/*! \tparam Item A fixed-size type with the size either 1 byte or a multiple of 8 bytes.
 *  \param in Input stream for loading the data.
 *  \return The loaded vector.
 *
 *  \par This corresponds to a vector of serializable items or bytes in simple-sds.
 */
template<typename Item>
std::vector<Item> load_vector(std::istream& in)
{
    static_assert(sizeof(Item) == 1 || sizeof(Item) % sizeof(element_type) == 0, "The size of an item must be 1 byte or a multiple of 8 bytes");
    std::vector<Item> result(load_value<size_t>(in));
    load_data(reinterpret_cast<char*>(result.data()), result.size() * sizeof(Item), in);
    return result;
}

//! Size of a vector in elements.
/*! \tparam Item A fixed-size type with the size either 1 byte or a multiple of 8 bytes.
 *  \param value A vector of items.
 *  \return Number of elements needed for serializing the vector.
 *
 *  \par This corresponds to a vector of serializable items or bytes in simple-sds.
 */
template<typename Item>
size_t vector_size(const std::vector<Item>& value)
{
    static_assert(sizeof(Item) == 1 || sizeof(Item) % sizeof(element_type) == 0, "The size of an item must be 1 byte or a multiple of 8 bytes");
    return value_size(value.size()) + data_size(value.size() * sizeof(Item));
}

//! Size of a vector in elements.
/*! \tparam Item A fixed-size type with the size either 1 byte or a multiple of 8 bytes.
 *  \param n Length of the vector.
 *  \return Number of elements needed for serializing the vector.
 *
 *  \par This corresponds to a vector of serializable items or bytes in simple-sds.
 */
template<typename Item>
size_t vector_size(size_t n)
{
    static_assert(sizeof(Item) == 1 || sizeof(Item) % sizeof(element_type) == 0, "The size of an item must be 1 byte or a multiple of 8 bytes");
    return value_size<size_t>() + data_size(n * sizeof(Item));
}

//-----------------------------------------------------------------------------

//! Serialize a string.
/*! \param value The string to be serialized.
 *  \param out Output stream for serialization.
 *
 *  \par This corresponds to a string in simple-sds.
 */
void serialize_string(const std::string& value, std::ostream& out);

//! Load a string.
/*! \param int Input stream for loading the data.
 *  \return The loaded string.
 *
 *  \par This corresponds to a string in simple-sds.
 *  However, the method does not validate that the bytes encode a valid UTF-8 string.
 */
std::string load_string(std::istream& in);

//! Size of a string in elements.
/*! \param value A string.
 *  \return Number of elements needed for serializing the string.
 *
 *  \par This corresponds to a string in simple-sds.
 */
size_t string_size(const std::string& value);

//! Size of a string in elements.
/*! \param n Length of the string.
 *  \return Number of elements needed for serializing the string.
 *
 *  \par This corresponds to a string in simple-sds.
 */
size_t string_size(size_t n);

//-----------------------------------------------------------------------------

//! Serialize an empty optional structure.
/*! \param out Output stream for serialization.
 *
 *  \par This corresponds to an absent optional structure in simple-sds.
 */
void empty_option(std::ostream& out);

//! Skip an optional structure without reading it.
/*! \param in Input stream for loading the data.
 *
 *  \par This corresponds to an absent optional structure in simple-sds.
 */
void skip_option(std::istream& in);

//! Size of an empty optional structure in elements.
/*! \return Number of elements needed for serializing an absent optional structure.
 *
 *  \par This corresponds to an absent optional structure in simple-sds.
 */
size_t empty_option_size();

//! Serialize a non-empty optional structure.
/*! \tparam Serialize A type implementing the serialization interface.
 *  \param value The value to be serialized.
 *  \param out Output stream for serialization.
 *
 *  \par This corresponds to a present optional structure in simple-sds.
 */
template<typename Serialize>
void serialize_option(const Serialize& value, std::ostream& out)
{
    serialize_value(value.simple_sds_size(), out);
    value.simple_sds_serialize(out);
}

//! Load an optional structure.
/*! \tparam Item A fixed-size type with the size either 1 byte or a multiple of 8 bytes.
 *  \param value The structure to be loaded.
 *  \param int Input stream for loading the data.
 *  \return Returns `true` if the structure was present and `false` if it was absent.
 *
 *  \par This corresponds to a present optional structure in simple-sds.
 *  If the structure is absent, `value` not modified.
 *  The method throws `InvalidData` if the structure is present but its size is invalid.
 */
template<typename Serialize>
bool load_option(Serialize& value, std::istream& in)
{
    size_t size = load_value<size_t>(in);
    if (size == 0) {
        return false;
    } else {
        std::streampos offset = in.tellg();
        size_t expected = static_cast<size_t>(offset) + size * sizeof(element_type);
        value.simple_sds_load(in);
        // Only do the sanity check if we got a valid starting offset.
        if (offset != -1 && static_cast<size_t>(in.tellg()) != expected) {
            throw InvalidData("Incorrect size for an optional structure");
        }
        return true;
    }
}

//! Size of an optional structure in elements.
/*! \param value An optional structure.
 *  \return Number of elements needed for serializing the optional structure.
 *
 *  \par This corresponds to a present optional structure in simple-sds.
 */
template<typename Serialize>
size_t option_size(const Serialize& value)
{
    return value_size<size_t>() + value.simple_sds_size();
}

//-----------------------------------------------------------------------------

//! Serializes a structure into the given file.
/*! \tparam Serialize A type implementing the serialization interface.
 *  \param data The structure to be serialized.
 *  \param filename File name for serialization.
 *
 *  \par If the file exists, it will be overwritten.
 *  Throws `CannotOpenFile` if the file cannot be opened.
 *  Output stream errors are thrown as exceptions.
 *  Any exceptions from serialization methods will be passed through.
 */
template<typename Serialize>
void serialize_to(const Serialize& data, const std::string& filename)
{
    // The default error message can be uninformative.
    std::ofstream out(filename, std::ios_base::binary);
    if (!out) {
        throw CannotOpenFile(filename, true);
    }

    out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    data.simple_sds_serialize(out);
    out.close();
}

//! Loads a structure from the given file.
/*! \tparam Serialize A type implementing the serialization interface.
 *  \param data The structure to be loaded.
 *  \param filename File name for the data.
 *
 *  \par Throws `CannotOpenFile` if the file cannot be opened.
 *  Input stream errors are thrown as exceptions.
 *  Any exceptions from serialization methods will be passed through.
 */
template<typename Serialize>
void load_from(Serialize& data, const std::string& filename)
{
    // The default error message can be uninformative.
    std::ifstream in(filename, std::ios_base::binary);
    if (!in) {
        throw CannotOpenFile(filename, false);
    }

    in.exceptions(std::ifstream::eofbit | std::ifstream::badbit | std::ifstream::failbit);
    data.simple_sds_load(in);
    in.close();
}

//-----------------------------------------------------------------------------

} // namespace simple_sds

} // namespace sdsl

#endif // INCLUDED_SDSL_SIMPLE_SDS
