#pragma once

#include "common.hpp"

namespace bjdata_serpent {

struct encode_options {
    /** As quick to write and read as can be, rather than as small: serpent's prefer::speed. */
    bool speed = false;
    bool sort_keys = false;
    /** Called with what nothing else can encode; returns something that can be. */
    nb::handle fallback {};
    /** Lists of two or more like dicts, and structured numpy arrays, as Structure-of-Arrays tables. */
    bool tables = false;
    /** Tables laid out field by field, `{${`, rather than record by record, `[${`. */
    bool by_column = false;
};

[[nodiscard]] nb::bytes encode(nb::handle value, const encode_options &options);

} // namespace bjdata_serpent
