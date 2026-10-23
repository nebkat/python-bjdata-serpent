#pragma once

#include "common.hpp"

#include <cstddef>
#include <span>

namespace bjdata_serpent {

struct decode_options {
    /** Packed, N-D and table arrays as numpy arrays, rather than (nested) lists. */
    bool numpy = false;
    nb::handle object_hook {};
    nb::handle object_pairs_hook {};
};

/** The value a whole document holds; DecodeError when it is not one well-formed value. */
[[nodiscard]] nb::object decode(std::span<const std::byte> document, const decode_options &options);

/**
 * The value the document starts with, and how many bytes it took: for a buffer holding one
 * value followed by anything else, such as the next.
 */
[[nodiscard]] nb::tuple decode_prefix(std::span<const std::byte> document, const decode_options &options);

/** DecodeError when the document is not one well-formed value. */
void validate(std::span<const std::byte> document);

} // namespace bjdata_serpent
