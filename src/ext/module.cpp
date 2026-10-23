#include "common.hpp"
#include "decode.hpp"
#include "encode.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <serpent/bjdata/json.hpp>
#include <serpent/bjdata/notation.hpp>
#include <serpent/bjdata/reader.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using namespace bjdata_serpent;

/** Calls `body` with a bytes-like object's bytes, held for as long as it runs. */
template<typename Body>
auto with_bytes(nb::handle data, Body body) {
    if (PyUnicode_Check(data.ptr())) {
        PyErr_SetString(PyExc_TypeError, "a BJData document is bytes, not str");
        propagate();
    }
    const buffer_view view { data.ptr(), PyBUF_SIMPLE };
    return body(std::span<const std::byte> { static_cast<const std::byte *>(view->buf), static_cast<std::size_t>(view->len) });
}

[[nodiscard]] decode_options decoding(bool numpy, nb::handle object_hook, nb::handle object_pairs_hook) {
    decode_options options;
    options.numpy = numpy;
    if (!object_hook.is_none()) options.object_hook = object_hook;
    if (!object_pairs_hook.is_none()) options.object_pairs_hook = object_pairs_hook;
    return options;
}

[[nodiscard]] bool choice(std::string_view given, std::string_view first, std::string_view second, const char *what) {
    if (given == first) return false;
    if (given == second) return true;
    PyErr_Format(PyExc_ValueError, "%s must be '%s' or '%s', not '%.*s'", what, std::string { first }.c_str(),
            std::string { second }.c_str(), static_cast<int>(given.size()), given.data());
    propagate();
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "The extension bjdata_serpent is built on; use the package, not this.";

    m.def(
            "encode",
            [](nb::handle value, std::string_view prefer, bool sort_keys, nb::handle fallback, bool tables, std::string_view table_order) {
                encode_options options;
                options.speed = choice(prefer, "size", "speed", "prefer");
                options.sort_keys = sort_keys;
                if (!fallback.is_none()) options.fallback = fallback;
                options.tables = tables;
                options.by_column = choice(table_order, "row", "column", "table_order");
                return encode(value, options);
            },
            "obj"_a.none(), nb::kw_only(), "prefer"_a = "size", "sort_keys"_a = false, "default"_a.none() = nb::none(), "tables"_a = false,
            "table_order"_a = "row");

    m.def(
            "decode",
            [](nb::handle data, bool numpy, nb::handle object_hook, nb::handle object_pairs_hook) {
                const auto options = decoding(numpy, object_hook, object_pairs_hook);
                return with_bytes(data, [&](std::span<const std::byte> document) { return decode(document, options); });
            },
            "data"_a, nb::kw_only(), "numpy"_a = false, "object_hook"_a.none() = nb::none(), "object_pairs_hook"_a.none() = nb::none());

    m.def(
            "decode_prefix",
            [](nb::handle data, bool numpy, nb::handle object_hook, nb::handle object_pairs_hook) {
                const auto options = decoding(numpy, object_hook, object_pairs_hook);
                return with_bytes(data, [&](std::span<const std::byte> document) { return decode_prefix(document, options); });
            },
            "data"_a, nb::kw_only(), "numpy"_a = false, "object_hook"_a.none() = nb::none(), "object_pairs_hook"_a.none() = nb::none());

    m.def(
            "validate", [](nb::handle data) { with_bytes(data, [](std::span<const std::byte> document) { validate(document); }); },
            "data"_a);

    m.def(
            "notation",
            [](nb::handle data) {
                return with_bytes(data, [](std::span<const std::byte> document) {
                    validate(document);
                    return serpent::bjdata::block_notation(document);
                });
            },
            "data"_a);

    m.def(
            "to_json",
            [](nb::handle data, std::size_t indent) {
                return with_bytes(data, [&](std::span<const std::byte> document) {
                    validate(document);
                    return serpent::json::encode(serpent::bjdata::reader::over(document), { .indent = indent });
                });
            },
            "data"_a, nb::kw_only(), "indent"_a = 0);
}
