#include "encode.hpp"

#include <serpent/bjdata/marker.hpp>
#include <serpent/bjdata/soa.hpp>
#include <serpent/bjdata/writer.hpp>
#include <serpent/limits.hpp>
#include <serpent/sink.hpp>

#include <nonstd/unaligned_ptr.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bjdata_serpent {

namespace {

using serpent::bjdata::basic_writer;
using serpent::bjdata::marker;
using serpent::bjdata::prefer;
using serpent::bjdata::soa::field_kind;

[[nodiscard]] std::string_view utf8(PyObject *text) {
    Py_ssize_t size = 0;
    const char *const data = PyUnicode_AsUTF8AndSize(text, &size);
    if (data == nullptr) propagate();
    return { data, static_cast<std::size_t>(size) };
}

[[nodiscard]] std::string_view bytes_of(PyObject *bytes) {
    char *data = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(bytes, &data, &size) < 0) propagate();
    return { data, static_cast<std::size_t>(size) };
}

[[noreturn]] void raise_type_error(const char *format, PyObject *object) {
    PyErr_Format(PyExc_TypeError, format, Py_TYPE(object)->tp_name);
    propagate();
}

[[nodiscard]] nb::object attribute(nb::handle object, const char *name) {
    return owned(PyObject_GetAttrString(object.ptr(), name));
}

[[nodiscard]] marker unsigned_marker_for(std::uint64_t most) noexcept {
    return serpent::bjdata::integer_marker(most);
}

[[nodiscard]] marker signed_marker(std::size_t width) noexcept {
    return width == 1 ? marker::int8 : width == 2 ? marker::int16 : width == 4 ? marker::int32 : width == 8 ? marker::int64 : marker::invalid;
}

[[nodiscard]] marker unsigned_marker(std::size_t width) noexcept {
    return width == 1 ? marker::uint8 : width == 2 ? marker::uint16 : width == 4 ? marker::uint32 : width == 8 ? marker::uint64 : marker::invalid;
}

[[nodiscard]] marker float_marker_of(std::size_t width) noexcept {
    return width == 2 ? marker::float16 : width == 4 ? marker::float32 : width == 8 ? marker::float64 : marker::invalid;
}

// ---------------- buffers ----------------

/** What a buffer's elements are, read from its struct-module format. */
struct element_format {
    marker type = marker::invalid;
    bool boolean = false;
    bool big_endian = false; ///< the bytes are not in BJData's order
    bool bytes = false;      ///< `B` or `c`: as bytes, from anything but a numpy array
};

/** A single number per element, or nothing: a structure, a repeat count, text, a pointer. */
[[nodiscard]] std::optional<element_format> format_of(const Py_buffer &view) noexcept {
    const char *format = view.format == nullptr ? "B" : view.format;
    char order = '@';
    if (*format == '@' || *format == '=' || *format == '<' || *format == '>' || *format == '!') order = *format++;
    const char code = *format++;
    if (code == '\0' || *format != '\0') return std::nullopt;

    element_format result;
    result.big_endian = order == '>' || order == '!' || ((order == '@' || order == '=') && std::endian::native == std::endian::big);
    const auto width = static_cast<std::size_t>(view.itemsize);
    switch (code) {
    case 'b':
    case 'h':
    case 'i':
    case 'l':
    case 'q':
    case 'n': result.type = signed_marker(width); break;
    case 'B':
    case 'c':
        result.bytes = true;
        [[fallthrough]];
    case 'H':
    case 'I':
    case 'L':
    case 'Q':
    case 'N': result.type = unsigned_marker(width); break;
    case 'e':
    case 'f':
    case 'd': result.type = float_marker_of(width); break;
    case '?':
        result.boolean = true;
        result.type = marker::uint8;
        break;
    default: return std::nullopt;
    }
    if (result.type == marker::invalid) return std::nullopt;
    return result;
}

/** An element's bits, as a little-endian word, from bytes in the given order. */
[[nodiscard]] std::uint64_t element_bits(const std::byte *at, std::size_t width, bool big_endian) noexcept {
    std::uint64_t bits = 0;
    for (std::size_t index = 0; index < width; ++index) {
        const auto byte = static_cast<std::uint64_t>(at[big_endian ? width - 1 - index : index]);
        bits |= byte << (index * 8);
    }
    return bits;
}

// ---------------- tables ----------------

/**
 * One field of a table's schema, with what it was decided from: for text, every record's value,
 * from which it is carried as characters, a dictionary or a table of offsets.
 */
struct column {
    PyObject *name = nullptr; ///< borrowed from the records or the dtype; null for an element of a run
    std::string_view label;
    field_kind kind = field_kind::null;
    marker type = marker::null;
    std::vector<column> children;

    // Text, until settled: a value a record, the distinct ones up to the bound, and their bytes.
    static constexpr std::size_t dictionary_bound = 255;
    std::vector<std::string_view> texts;
    std::vector<std::string_view> entries;
    std::unordered_map<std::string_view, std::uint32_t> index;
    std::uint64_t text_bytes = 0;
    bool characters = true;
    bool overflowed = false;

    void note(std::string_view text) {
        this->texts.push_back(text);
        this->text_bytes += text.size();
        if (text.size() != 1 || static_cast<unsigned char>(text[0]) > 127) this->characters = false;
        if (this->overflowed || this->index.contains(text)) return;
        if (this->entries.size() == dictionary_bound) {
            this->overflowed = true;
            return;
        }
        this->index.emplace(text, static_cast<std::uint32_t>(this->entries.size()));
        this->entries.push_back(text);
    }

    /** Text decides how it is carried once every record's value has been noted. */
    void settle() {
        if (this->kind == field_kind::text) {
            if (this->characters && !this->texts.empty()) {
                this->kind = field_kind::scalar;
                this->type = marker::character;
            } else if (this->overflowed) {
                // Each record stores its own index into the offsets, which hold the text's length.
                this->kind = field_kind::offsets;
                this->type = unsigned_marker_for(std::max<std::uint64_t>(this->text_bytes, this->texts.size()));
            } else {
                this->kind = field_kind::dictionary;
                this->type = unsigned_marker_for(this->entries.size());
            }
        }
        for (auto &child : this->children) child.settle();
    }

    [[nodiscard]] std::size_t width() const noexcept {
        switch (this->kind) {
        case field_kind::boolean: return 1;
        case field_kind::null: return 0;
        case field_kind::record:
        case field_kind::array: {
            std::size_t total = 0;
            for (const auto &child : this->children) total += child.width();
            return total;
        }
        default: return serpent::bjdata::payload_width(this->type);
        }
    }
};

[[nodiscard]] bool is_bool(PyObject *value) noexcept { return value == Py_True || value == Py_False; }

std::optional<std::vector<column>> infer_record(std::span<PyObject *const> records, bool speed);

/** A column from one key's value in every record, when they are all of one kind a schema can say. */
[[nodiscard]] std::optional<column> infer_column(PyObject *name, std::span<PyObject *const> values, bool speed) {
    column result;
    result.name = name;
    if (name != nullptr) result.label = utf8(name);
    PyObject *const first = values.front();

    if (is_bool(first)) {
        if (!std::ranges::all_of(values, is_bool)) return std::nullopt;
        result.kind = field_kind::boolean;
        return result;
    }
    if (first == Py_None) {
        if (!std::ranges::all_of(values, [](PyObject *value) { return value == Py_None; })) return std::nullopt;
        result.kind = field_kind::null;
        return result;
    }
    if ((PyLong_Check(first) && !is_bool(first)) || PyFloat_Check(first)) {
        bool real = false;
        bool fit_half = true;
        bool fit_single = true;
        bool beyond_signed = false;
        std::int64_t least = 0;
        std::int64_t most = 0;
        std::uint64_t most_unsigned = 0;
        bool seen = false;
        for (PyObject *const value : values) {
            if (PyFloat_Check(value)) {
                real = true;
                const double number = PyFloat_AS_DOUBLE(value);
                fit_half = fit_half && serpent::bjdata::fits_float16(number);
                fit_single = fit_single && serpent::bjdata::fits_float32(number);
                continue;
            }
            if (!PyLong_Check(value) || is_bool(value)) return std::nullopt;
            int overflow = 0;
            const long long whole = PyLong_AsLongLongAndOverflow(value, &overflow);
            if (whole == -1 && PyErr_Occurred()) propagate();
            if (overflow < 0) return std::nullopt;
            if (overflow > 0) {
                const unsigned long long wide = PyLong_AsUnsignedLongLong(value);
                if (wide == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
                    PyErr_Clear();
                    return std::nullopt;
                }
                beyond_signed = true;
                most_unsigned = std::max<std::uint64_t>(most_unsigned, wide);
                continue;
            }
            least = seen ? std::min<std::int64_t>(least, whole) : whole;
            most = seen ? std::max<std::int64_t>(most, whole) : whole;
            seen = true;
        }
        result.kind = field_kind::scalar;
        if (real) {
            // A column with any real in it is reals, which every integer in it must be exactly.
            constexpr std::int64_t exact = std::int64_t { 1 } << 53;
            if (beyond_signed || least < -exact || most > exact) return std::nullopt;
            if (seen) {
                fit_half = fit_half && serpent::bjdata::fits_float16(static_cast<double>(least)) && serpent::bjdata::fits_float16(static_cast<double>(most));
                fit_single = fit_single && serpent::bjdata::fits_float32(static_cast<double>(least)) && serpent::bjdata::fits_float32(static_cast<double>(most));
            }
            // The narrowest that holds every value exactly where size is preferred, as for a real on its own.
            result.type = speed ? marker::float64 : serpent::bjdata::float_marker(fit_half, fit_single);
        } else if (beyond_signed) {
            if (seen && least < 0) return std::nullopt;
            result.type = speed ? marker::uint64 : unsigned_marker_for(std::max<std::uint64_t>(most_unsigned, static_cast<std::uint64_t>(most)));
        } else {
            result.type = speed ? marker::int64 : serpent::bjdata::integer_marker(least, most);
        }
        return result;
    }
    if (PyUnicode_Check(first)) {
        result.kind = field_kind::text;
        for (PyObject *const value : values) {
            if (!PyUnicode_Check(value)) return std::nullopt;
            result.note(utf8(value));
        }
        return result;
    }
    if (PyList_Check(first) || PyTuple_Check(first)) {
        // Lists of one length in every record: a fixed run, each position a column of its own.
        const auto length = PySequence_Fast_GET_SIZE(first);
        if (length == 0) return std::nullopt;
        for (PyObject *const value : values) {
            if (Py_TYPE(value) != Py_TYPE(first) || PySequence_Fast_GET_SIZE(value) != length) return std::nullopt;
        }
        result.kind = field_kind::array;
        std::vector<PyObject *> elements(values.size());
        for (Py_ssize_t position = 0; position < length; ++position) {
            for (std::size_t index = 0; index < values.size(); ++index)
                elements[index] = PySequence_Fast_GET_ITEM(values[index], position);
            auto element = infer_column(nullptr, elements, speed);
            if (!element) return std::nullopt;
            result.children.push_back(std::move(*element));
        }
        return result;
    }
    if (PyDict_Check(first)) {
        auto fields = infer_record(values, speed);
        if (!fields) return std::nullopt;
        result.kind = field_kind::record;
        result.children = std::move(*fields);
        return result;
    }
    return std::nullopt;
}

/** The columns of records that are all dicts with the same str keys, each column of one kind. */
std::optional<std::vector<column>> infer_record(std::span<PyObject *const> records, bool speed) {
    PyObject *const first = records.front();
    const auto size = PyDict_Size(first);
    if (size == 0) return std::nullopt;
    for (PyObject *const record : records) {
        if (!PyDict_Check(record) || PyDict_Size(record) != size) return std::nullopt;
    }

    std::vector<column> fields;
    std::vector<PyObject *> values(records.size());
    Py_ssize_t position = 0;
    PyObject *key = nullptr;
    PyObject *ignored = nullptr;
    while (PyDict_Next(first, &position, &key, &ignored)) {
        if (!PyUnicode_Check(key)) return std::nullopt;
        for (std::size_t index = 0; index < records.size(); ++index) {
            values[index] = PyDict_GetItemWithError(records[index], key);
            if (values[index] == nullptr) {
                if (PyErr_Occurred()) propagate();
                return std::nullopt;
            }
        }
        auto field = infer_column(key, values, speed);
        if (!field) return std::nullopt;
        fields.push_back(std::move(*field));
    }
    return fields;
}

/** A column from a numpy dtype, or nothing for a dtype a schema cannot say. */
[[nodiscard]] std::optional<column> dtype_column(PyObject *name, nb::handle dtype) {
    column result;
    result.name = name;
    if (name != nullptr) result.label = utf8(name);

    const auto names = attribute(dtype, "names");
    if (!names.is_none()) {
        const auto fields = attribute(dtype, "fields");
        result.kind = field_kind::record;
        for (const auto field_name : names) {
            const auto described = owned(PyObject_GetItem(fields.ptr(), field_name.ptr()));
            auto child = dtype_column(field_name.ptr(), nb::borrow(PyTuple_GetItem(described.ptr(), 0)));
            if (!child) return std::nullopt;
            result.children.push_back(std::move(*child));
        }
        return result;
    }
    const auto sub = attribute(dtype, "subdtype");
    if (!sub.is_none()) {
        const auto base = nb::borrow(PyTuple_GetItem(sub.ptr(), 0));
        const auto shape = nb::borrow(PyTuple_GetItem(sub.ptr(), 1));
        if (PyTuple_Size(shape.ptr()) != 1) return std::nullopt;
        const auto count = PyLong_AsSsize_t(PyTuple_GetItem(shape.ptr(), 0));
        if (count <= 0) return std::nullopt;
        result.kind = field_kind::array;
        for (Py_ssize_t index = 0; index < count; ++index) {
            auto element = dtype_column(nullptr, base);
            if (!element) return std::nullopt;
            result.children.push_back(std::move(*element));
        }
        return result;
    }

    const auto kind = utf8(attribute(dtype, "kind").ptr());
    const auto width = static_cast<std::size_t>(PyLong_AsSsize_t(attribute(dtype, "itemsize").ptr()));
    switch (kind.empty() ? '\0' : kind[0]) {
    case 'b': result.kind = field_kind::boolean; break;
    case 'i':
        result.kind = field_kind::scalar;
        result.type = signed_marker(width);
        break;
    case 'u':
        result.kind = field_kind::scalar;
        result.type = unsigned_marker(width);
        break;
    case 'f':
        result.kind = field_kind::scalar;
        result.type = float_marker_of(width);
        break;
    case 'U':
    case 'S': result.kind = field_kind::text; break;
    default: return std::nullopt;
    }
    if (result.kind == field_kind::scalar && result.type == marker::invalid) return std::nullopt;
    return result;
}

/** A field's value in its record: by key in a dict, by position in a tuple or list. */
[[nodiscard]] PyObject *field_value(PyObject *record, const column &field, std::size_t position) {
    if (PyDict_Check(record)) {
        PyObject *const value = PyDict_GetItemWithError(record, field.name);
        if (value == nullptr) {
            if (PyErr_Occurred()) propagate();
            raise_encode_error("a record of a table lost a key while it was being written");
        }
        return value;
    }
    if (PyTuple_Check(record) && static_cast<std::size_t>(PyTuple_GET_SIZE(record)) > position)
        return PyTuple_GET_ITEM(record, static_cast<Py_ssize_t>(position));
    if (PyList_Check(record) && static_cast<std::size_t>(PyList_GET_SIZE(record)) > position)
        return PyList_GET_ITEM(record, static_cast<Py_ssize_t>(position));
    raise_encode_error("a record of a table does not have the fields its schema says");
}

/**
 * A record of a structured array as plain tuples and lists all the way down: numpy's tolist()
 * leaves a subarray field as a small array of its own.
 */
[[nodiscard]] nb::object plain(PyObject *value) {
    if (PyTuple_Check(value) || PyList_Check(value)) {
        const bool tuple = PyTuple_Check(value);
        const auto size = PySequence_Fast_GET_SIZE(value);
        auto result = owned(tuple ? PyTuple_New(size) : PyList_New(size));
        for (Py_ssize_t index = 0; index < size; ++index) {
            PyObject *const item = plain(PySequence_Fast_GET_ITEM(value, index)).release().ptr();
            if (tuple) PyTuple_SET_ITEM(result.ptr(), index, item);
            else PyList_SET_ITEM(result.ptr(), index, item);
        }
        return result;
    }
    if (!PyUnicode_Check(value) && !PyBytes_Check(value) && PyObject_HasAttrString(value, "tolist")) {
        const auto list = owned(PyObject_CallMethod(value, "tolist", nullptr));
        return plain(list.ptr());
    }
    return nb::borrow(value);
}

/** Notes the text of one record's value of a column read from a dtype. */
void observe(column &field, PyObject *value) {
    switch (field.kind) {
    case field_kind::text:
        if (PyUnicode_Check(value)) field.note(utf8(value));
        else if (PyBytes_Check(value)) field.note(bytes_of(value));
        else raise_type_error("a text field of a structured array holds a %s", value);
        break;
    case field_kind::record:
    case field_kind::array:
        for (std::size_t index = 0; index < field.children.size(); ++index)
            observe(field.children[index], field_value(value, field.children[index], index));
        break;
    default: break;
    }
}

template<typename T>
void store(std::string &into, T value) {
    const nonstd::unaligned_little<T> stored { value };
    into.append(reinterpret_cast<const char *>(stored.data()), nonstd::unaligned_little<T>::storage_bytes);
}

void store_unsigned(std::string &into, marker width, std::uint64_t value) {
    switch (width) {
    case marker::uint8: store(into, static_cast<std::uint8_t>(value)); break;
    case marker::uint16: store(into, static_cast<std::uint16_t>(value)); break;
    case marker::uint32: store(into, static_cast<std::uint32_t>(value)); break;
    default: store(into, value); break;
    }
}

/** One record's bytes of a field: no marker, no key, only the value at the schema's width. */
void store_field(std::string &into, const column &field, PyObject *value, std::size_t record) {
    switch (field.kind) {
    case field_kind::null: return;
    case field_kind::boolean:
        into.push_back(static_cast<char>(PyObject_IsTrue(value) == 1 ? marker::boolean_true : marker::boolean_false));
        return;
    case field_kind::dictionary: store_unsigned(into, field.type, field.index.at(field.texts[record])); return;
    case field_kind::offsets: store_unsigned(into, field.type, record); return;
    case field_kind::record:
    case field_kind::array:
        for (std::size_t index = 0; index < field.children.size(); ++index)
            store_field(into, field.children[index], field_value(value, field.children[index], index), record);
        return;
    case field_kind::scalar: break;
    default: raise_encode_error("a field a table cannot hold");
    }

    switch (field.type) {
    case marker::character: into.push_back(field.texts[record][0]); return;
    case marker::float16:
    case marker::float32:
    case marker::float64: {
        const double real = PyFloat_AsDouble(value);
        if (real == -1.0 && PyErr_Occurred()) propagate();
        if (field.type == marker::float16) store(into, serpent::bjdata::encode_float16(static_cast<float>(real)));
        else if (field.type == marker::float32) store(into, static_cast<float>(real));
        else store(into, real);
        return;
    }
    case marker::uint64: {
        const unsigned long long whole = PyLong_AsUnsignedLongLong(value);
        if (whole == static_cast<unsigned long long>(-1) && PyErr_Occurred()) propagate();
        store(into, static_cast<std::uint64_t>(whole));
        return;
    }
    default: break;
    }
    const long long whole = PyLong_AsLongLong(value);
    if (whole == -1 && PyErr_Occurred()) propagate();
    switch (field.type) {
    case marker::uint8: store(into, static_cast<std::uint8_t>(whole)); break;
    case marker::int8: store(into, static_cast<std::int8_t>(whole)); break;
    case marker::uint16: store(into, static_cast<std::uint16_t>(whole)); break;
    case marker::int16: store(into, static_cast<std::int16_t>(whole)); break;
    case marker::uint32: store(into, static_cast<std::uint32_t>(whole)); break;
    case marker::int32: store(into, static_cast<std::int32_t>(whole)); break;
    default: store(into, static_cast<std::int64_t>(whole)); break;
    }
}

/** A table: its records, what their fields are, and how many there are in which shape. */
struct table {
    std::vector<PyObject *> records; ///< borrowed from a list the caller holds
    column layout;                   ///< a record
    std::vector<std::uint64_t> shape; ///< more than one extent for an N-D numpy array
};

// ---------------- the encoder ----------------

template<prefer Preference>
class encoder {
    using writer = basic_writer<Preference>;

    writer &out;
    const encode_options &options;
    int depth = 0;

    PyObject *ndarray_type = nullptr; ///< numpy's, once something has imported numpy
    PyObject *generic_type = nullptr;
    PyObject *decimal = nullptr;

    /** Counts a level of nesting - a container, or a call to `default` - for as long as it lives. */
    class level {
        encoder &self;

    public:
        explicit level(encoder &self) : self(self) {
            if (++self.depth > serpent::max_depth)
                raise_encode_error("nesting deeper than " + std::to_string(serpent::max_depth) + " levels, or a value that contains itself");
        }
        level(const level &) = delete;
        level &operator=(const level &) = delete;
        ~level() { --this->self.depth; }
    };

    static constexpr bool speed = Preference == prefer::speed;

    void integer(PyObject *value) {
        int overflow = 0;
        const long long whole = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (whole == -1 && PyErr_Occurred()) propagate();
        if (overflow == 0) {
            this->out.integer(static_cast<std::int64_t>(whole));
            return;
        }
        if (overflow > 0) {
            const unsigned long long wide = PyLong_AsUnsignedLongLong(value);
            if (!(wide == static_cast<unsigned long long>(-1) && PyErr_Occurred())) {
                this->out.integer(static_cast<std::uint64_t>(wide));
                return;
            }
            PyErr_Clear();
        }
        // Too wide for any integer marker: its decimal digits.
        const auto digits = owned(PyObject_Str(value));
        this->out.high_precision(utf8(digits.ptr()));
    }

    void key(PyObject *name) {
        if (!PyUnicode_Check(name)) raise_type_error("keys must be str, not %s", name);
        this->out.key(utf8(name));
    }

    void sequence(PyObject *items) {
        const level nested { *this };
        if (this->options.tables && PyList_Check(items) && this->try_dict_table(items)) return;

        // A list can be changed by the `default` called for one of its elements, so its length
        // is looked at again each time round, and a strong reference held to each element.
        const auto length = [&] { return PyList_Check(items) ? PyList_GET_SIZE(items) : PyTuple_GET_SIZE(items); };
        const auto at = [&](Py_ssize_t index) {
            return nb::borrow(PyList_Check(items) ? PyList_GET_ITEM(items, index) : PyTuple_GET_ITEM(items, index));
        };
        if constexpr (speed) {
            const auto count = length();
            const auto scope = this->out.counted_array(static_cast<std::uint64_t>(count));
            for (Py_ssize_t index = 0; index < count; ++index) {
                if (length() != count) {
                    PyErr_SetString(PyExc_RuntimeError, "list changed size during encoding");
                    propagate();
                }
                this->value(at(index).ptr());
            }
        } else {
            const auto scope = this->out.array();
            for (Py_ssize_t index = 0; index < length(); ++index) this->value(at(index).ptr());
        }
    }

    void mapping(PyObject *map) {
        const level nested { *this };
        const auto items = owned(PyDict_Check(map) ? PyDict_Items(map) : PyMapping_Items(map));
        const auto count = PyList_GET_SIZE(items.ptr());
        for (Py_ssize_t index = 0; index < count; ++index) {
            PyObject *const name = PyTuple_GET_ITEM(PyList_GET_ITEM(items.ptr(), index), 0);
            if (!PyUnicode_Check(name)) raise_type_error("keys must be str, not %s", name);
        }
        if (this->options.sort_keys && PyList_Sort(items.ptr()) < 0) propagate();

        const auto members = [&] {
            for (Py_ssize_t index = 0; index < count; ++index) {
                PyObject *const pair = PyList_GET_ITEM(items.ptr(), index);
                this->key(PyTuple_GET_ITEM(pair, 0));
                this->value(PyTuple_GET_ITEM(pair, 1));
            }
        };
        if constexpr (speed) {
            const auto scope = this->out.counted_object(static_cast<std::uint64_t>(count));
            members();
        } else {
            const auto scope = this->out.object();
            members();
        }
    }

    /** A number held in a buffer of one element, at its own type when speed is preferred. */
    void buffered_scalar(const element_format &format, const std::byte *at) {
        const auto width = serpent::bjdata::payload_width(format.type);
        const auto bits = element_bits(at, width, format.big_endian);
        if (format.boolean) {
            this->out.boolean(bits != 0);
            return;
        }
        switch (format.type) {
        case marker::uint8: this->out.number(static_cast<std::uint8_t>(bits)); break;
        case marker::int8: this->out.number(static_cast<std::int8_t>(bits)); break;
        case marker::uint16: this->out.number(static_cast<std::uint16_t>(bits)); break;
        case marker::int16: this->out.number(static_cast<std::int16_t>(bits)); break;
        case marker::uint32: this->out.number(static_cast<std::uint32_t>(bits)); break;
        case marker::int32: this->out.number(static_cast<std::int32_t>(bits)); break;
        case marker::uint64: this->out.number(static_cast<std::uint64_t>(bits)); break;
        case marker::int64: this->out.number(static_cast<std::int64_t>(bits)); break;
        case marker::float16:
            if constexpr (speed) {
                this->out.put_marker(marker::float16);
                this->out.put_raw(static_cast<std::uint16_t>(bits));
            } else {
                this->out.real(static_cast<double>(serpent::bjdata::decode_float16(static_cast<std::uint16_t>(bits))));
            }
            break;
        case marker::float32: this->out.number(std::bit_cast<float>(static_cast<std::uint32_t>(bits))); break;
        default: this->out.number(std::bit_cast<double>(bits)); break;
        }
    }

    /**
     * Anything with a buffer of numbers: a typed array, counted by its dimensions when it has
     * more than one, in one copy where its bytes are already in order. False for a buffer of
     * anything else, having written nothing.
     */
    [[nodiscard]] bool buffered(PyObject *object, bool from_numpy) {
        const buffer_view view { object, PyBUF_RECORDS_RO };
        const auto format = format_of(*view);
        if (!format) return false;
        if (view->ndim == 0) {
            this->buffered_scalar(*format, static_cast<const std::byte *>(view->buf));
            return true;
        }
        if (format->boolean) return false;
        if (static_cast<std::size_t>(view->ndim) > serpent::bjdata::max_dimensions)
            raise_encode_error("an array of more than " + std::to_string(serpent::bjdata::max_dimensions) + " dimensions");

        const auto length = static_cast<std::size_t>(view->len);
        const auto *data = static_cast<const std::byte *>(view->buf);
        std::vector<std::byte> copy;
        if (!PyBuffer_IsContiguous(&*view, 'C') || format->big_endian) {
            copy.resize(length);
            if (PyBuffer_ToContiguous(copy.data(), &*view, view->len, 'C') < 0) propagate();
            if (format->big_endian) {
                const auto width = static_cast<std::size_t>(view->itemsize);
                for (std::size_t at = 0; at + width <= length; at += width) std::reverse(copy.begin() + at, copy.begin() + at + width);
            }
            data = copy.data();
        }

        // Bytes from anything but numpy - a memoryview, an array('B') - are binary, as bytes are.
        if (format->bytes && !from_numpy && view->ndim == 1) {
            this->out.binary({ data, length });
            return true;
        }
        if (view->ndim == 1) {
            this->out.typed_header(format->type, static_cast<std::uint64_t>(view->shape[0]));
        } else {
            this->out.put_marker(marker::array_begin);
            this->out.put_marker(marker::strong_type);
            this->out.put_marker(format->type);
            this->out.put_marker(marker::count);
            this->out.put_marker(marker::array_begin);
            for (int dimension = 0; dimension < view->ndim; ++dimension)
                this->out.put_length(static_cast<std::uint64_t>(view->shape[dimension]));
            this->out.put_marker(marker::array_end);
        }
        this->out.put({ data, length });
        return true;
    }

    /** What the object turns into as a list: numpy's and array's tolist(). */
    void as_list(PyObject *object) {
        const auto list = owned(PyObject_CallMethod(object, "tolist", nullptr));
        this->value(list.ptr());
    }

    /** A record of a structured array, or an array of them, as objects named by its dtype. */
    void struct_value(PyObject *value, nb::handle dtype) {
        const auto names = attribute(dtype, "names");
        if (names.is_none()) {
            this->value(value);
            return;
        }
        const level nested { *this };
        if (PyList_Check(value)) {
            const auto scope = this->out.array();
            for (Py_ssize_t index = 0; index < PyList_GET_SIZE(value); ++index)
                this->struct_value(PyList_GET_ITEM(value, index), dtype);
            return;
        }
        const auto fields = attribute(dtype, "fields");
        const auto scope = this->out.object();
        Py_ssize_t position = 0;
        for (const auto name : names) {
            const auto described = owned(PyObject_GetItem(fields.ptr(), name.ptr()));
            this->key(name.ptr());
            this->struct_value(field_value(value, column {}, static_cast<std::size_t>(position++)),
                    nb::borrow(PyTuple_GetItem(described.ptr(), 0)));
        }
    }

    void numpy_array(PyObject *array) {
        const auto dtype = attribute(array, "dtype");
        if (!attribute(dtype, "names").is_none()) {
            if (this->options.tables && this->try_dtype_table(array, dtype)) return;
            const auto records = owned(PyObject_CallMethod(array, "tolist", nullptr));
            this->struct_value(records.ptr(), dtype);
            return;
        }
        const auto kind = utf8(attribute(dtype, "kind").ptr());
        if ((kind == "i" || kind == "u" || kind == "f") && this->buffered(array, true)) return;
        this->as_list(array);
    }

    void numpy_scalar(PyObject *scalar) {
        const auto dtype = attribute(scalar, "dtype");
        if (!attribute(dtype, "names").is_none()) {
            const auto record = owned(PyObject_CallMethod(scalar, "item", nullptr));
            this->struct_value(record.ptr(), dtype);
            return;
        }
        if (PyObject_CheckBuffer(scalar) && this->buffered(scalar, true)) return;
        this->as_list(scalar);
    }

    void other(PyObject *value) {
        if (this->ndarray_type == nullptr) {
            if (PyObject *const numpy = imported("numpy")) {
                this->ndarray_type = attribute(numpy, "ndarray").release().ptr();
                this->generic_type = attribute(numpy, "generic").release().ptr();
            }
        }
        if (this->ndarray_type != nullptr) {
            if (PyObject_TypeCheck(value, reinterpret_cast<PyTypeObject *>(this->ndarray_type))) {
                this->numpy_array(value);
                return;
            }
            if (PyObject_TypeCheck(value, reinterpret_cast<PyTypeObject *>(this->generic_type))) {
                this->numpy_scalar(value);
                return;
            }
        }
        if (this->decimal == nullptr && imported("decimal") != nullptr) this->decimal = decimal_type().ptr();
        if (this->decimal != nullptr && PyObject_TypeCheck(value, reinterpret_cast<PyTypeObject *>(this->decimal))) {
            const auto finite = owned(PyObject_CallMethod(value, "is_finite", nullptr));
            if (PyObject_IsTrue(finite.ptr()) == 1) {
                const auto digits = owned(PyObject_Str(value));
                this->out.high_precision(utf8(digits.ptr()));
            } else {
                const double real = PyFloat_AsDouble(owned(PyNumber_Float(value)).ptr());
                this->out.real(real);
            }
            return;
        }
        if (PyObject_CheckBuffer(value)) {
            if (this->buffered(value, false)) return;
            if (PyObject_HasAttrString(value, "tolist")) {
                this->as_list(value);
                return;
            }
        }
        if (PyMapping_Check(value) && PyObject_HasAttrString(value, "items")) {
            this->mapping(value);
            return;
        }
        if (PySequence_Check(value)) {
            const auto items = owned(PySequence_Tuple(value));
            this->sequence(items.ptr());
            return;
        }
        if (this->options.fallback.is_valid()) {
            const level nested { *this };
            const auto replaced = this->options.fallback(nb::handle(value));
            this->value(replaced.ptr());
            return;
        }
        raise_type_error("Object of type %s is not BJData serializable", value);
    }

    // ---------------- tables ----------------

    void write_name(std::string_view name) {
        this->out.put_length(name.size());
        this->out.put_text(name);
    }

    void write_type(const column &field) {
        switch (field.kind) {
        case field_kind::scalar: this->out.put_marker(field.type); break;
        case field_kind::boolean: this->out.put_marker(marker::boolean_true); break;
        case field_kind::null: this->out.put_marker(marker::null); break;
        case field_kind::dictionary:
            this->out.put_marker(marker::array_begin);
            this->out.put_marker(marker::strong_type);
            this->out.put_marker(marker::string);
            this->out.put_marker(marker::count);
            this->out.put_length(field.entries.size());
            for (const auto entry : field.entries) this->write_name(entry);
            break;
        case field_kind::offsets:
            this->out.put_marker(marker::array_begin);
            this->out.put_marker(marker::strong_type);
            this->out.put_marker(field.type);
            this->out.put_marker(marker::array_end);
            break;
        case field_kind::record:
            this->out.put_marker(marker::object_begin);
            for (const auto &child : field.children) {
                this->write_name(child.label);
                this->write_type(child);
            }
            this->out.put_marker(marker::object_end);
            break;
        case field_kind::array:
            this->out.put_marker(marker::array_begin);
            for (const auto &child : field.children) this->write_type(child);
            this->out.put_marker(marker::array_end);
            break;
        default: raise_encode_error("a field a table cannot hold");
        }
    }

    /** Each offset column's table of offsets, then its text, in schema order. */
    void write_offsets(const column &field) {
        if (field.kind == field_kind::offsets) {
            std::string offsets;
            std::uint64_t offset = 0;
            store_unsigned(offsets, field.type, 0);
            for (const auto text : field.texts) {
                offset += text.size();
                store_unsigned(offsets, field.type, offset);
            }
            this->out.put_text(offsets);
            for (const auto text : field.texts) this->out.put_text(text);
        }
        for (const auto &child : field.children) this->write_offsets(child);
    }

    void write_table(table &source) {
        source.layout.settle();
        const auto &fields = source.layout.children;

        this->out.put_marker(this->options.by_column ? marker::object_begin : marker::array_begin);
        this->out.put_marker(marker::strong_type);
        this->write_type(source.layout);
        this->out.put_marker(marker::count);
        if (source.shape.size() > 1) {
            this->out.put_marker(marker::array_begin);
            for (const auto extent : source.shape) this->out.put_length(extent);
            this->out.put_marker(marker::array_end);
        } else {
            this->out.put_length(source.records.size());
        }

        // The payload is gathered first: storing a field calls into Python, which may raise, and
        // nothing that may is called from inside the writer.
        std::string payload;
        payload.reserve(source.records.size() * source.layout.width());
        if (this->options.by_column) {
            for (std::size_t index = 0; index < fields.size(); ++index) {
                for (std::size_t record = 0; record < source.records.size(); ++record)
                    store_field(payload, fields[index], field_value(source.records[record], fields[index], index), record);
            }
        } else {
            for (std::size_t record = 0; record < source.records.size(); ++record) {
                for (std::size_t index = 0; index < fields.size(); ++index)
                    store_field(payload, fields[index], field_value(source.records[record], fields[index], index), record);
            }
        }
        this->out.put_text(payload);
        this->write_offsets(source.layout);
    }

    /** A list of two or more dicts alike enough for one schema, as a table; false if it is not one. */
    [[nodiscard]] bool try_dict_table(PyObject *list) {
        const auto count = PyList_GET_SIZE(list);
        if (count < 2 || !PyDict_Check(PyList_GET_ITEM(list, 0))) return false;
        table source;
        source.records.assign(&PyList_GET_ITEM(list, 0), &PyList_GET_ITEM(list, 0) + count);
        auto fields = infer_record(source.records, speed);
        if (!fields) return false;
        source.layout.kind = field_kind::record;
        source.layout.children = std::move(*fields);
        this->write_table(source);
        return true;
    }

    /** A structured numpy array of two or more records as a table; false if its dtype cannot be one. */
    [[nodiscard]] bool try_dtype_table(PyObject *array, nb::handle dtype) {
        const auto size = PyLong_AsSsize_t(attribute(array, "size").ptr());
        if (size < 2) return false;
        auto layout = dtype_column(nullptr, dtype);
        if (!layout) return false;

        const auto flat = owned(PyObject_CallMethod(array, "reshape", "i", -1));
        const auto rows = plain(owned(PyObject_CallMethod(flat.ptr(), "tolist", nullptr)).ptr());
        table source;
        source.layout = std::move(*layout);
        source.records.assign(&PyList_GET_ITEM(rows.ptr(), 0), &PyList_GET_ITEM(rows.ptr(), 0) + size);
        for (PyObject *const record : source.records) observe(source.layout, record);

        const auto shape = attribute(array, "shape");
        if (PyTuple_GET_SIZE(shape.ptr()) > 1) {
            for (const auto extent : shape) source.shape.push_back(PyLong_AsUnsignedLongLong(extent.ptr()));
        }
        this->write_table(source);
        return true;
    }

public:
    encoder(writer &out, const encode_options &options) noexcept : out(out), options(options) {}

    void value(PyObject *value) {
        if (value == Py_None) this->out.null();
        else if (value == Py_True) this->out.boolean(true);
        else if (value == Py_False) this->out.boolean(false);
        else if (PyLong_Check(value)) this->integer(value);
        else if (PyFloat_Check(value)) this->out.real(PyFloat_AS_DOUBLE(value));
        else if (PyUnicode_Check(value)) this->out.string(utf8(value));
        else if (PyBytes_Check(value))
            this->out.binary({ reinterpret_cast<const std::byte *>(PyBytes_AS_STRING(value)), static_cast<std::size_t>(PyBytes_GET_SIZE(value)) });
        else if (PyByteArray_Check(value))
            this->out.binary({ reinterpret_cast<const std::byte *>(PyByteArray_AsString(value)), static_cast<std::size_t>(PyByteArray_Size(value)) });
        else if (PyList_Check(value) || PyTuple_Check(value)) this->sequence(value);
        else if (PyDict_Check(value)) this->mapping(value);
        else this->other(value);
    }
};

/**
 * Writes the document straight into the bytes object that will be returned, lending its storage
 * to the writer, so the document is written once and never copied out of anything.
 *
 * The writer calls this from code that may not throw, so a failure to grow - a MemoryError, set
 * as it happens - is an empty lend, which the writer latches as a failed sink.
 */
class bytes_sink {
    static constexpr std::size_t slack = 256;

    PyObject *bytes = nullptr;
    std::size_t length = 0;   ///< bytes of the document so far
    std::size_t lent_at = 0;

    [[nodiscard]] std::size_t capacity() const noexcept { return this->bytes == nullptr ? 0 : static_cast<std::size_t>(PyBytes_GET_SIZE(this->bytes)); }
    [[nodiscard]] std::byte *data() const noexcept { return reinterpret_cast<std::byte *>(PyBytes_AS_STRING(this->bytes)); }

    [[nodiscard]] bool reserve(std::size_t size) noexcept {
        if (this->bytes == nullptr) return false;
        if (size <= this->capacity()) return true;
        const auto grown = std::max(size, this->capacity() * 2);
        return _PyBytes_Resize(&this->bytes, static_cast<Py_ssize_t>(grown)) == 0; // releases it on failure
    }

public:
    bytes_sink() noexcept : bytes(PyBytes_FromStringAndSize(nullptr, 256)) {}
    bytes_sink(const bytes_sink &) = delete;
    bytes_sink &operator=(const bytes_sink &) = delete;
    ~bytes_sink() { Py_XDECREF(this->bytes); }

    /** A payload long enough that the writer hands it over whole, with room left after it for what follows. */
    bool write(std::span<const std::byte> written) noexcept {
        if (!this->reserve(this->length + written.size() + slack)) return false;
        std::memcpy(this->data() + this->length, written.data(), written.size());
        this->length += written.size();
        return true;
    }

    /**
     * What room is left, when it is at least what has to land in one piece: growing the bytes may
     * move them, copying the whole document so far, so it waits until the room is used up.
     */
    [[nodiscard]] std::span<std::byte> lend(std::size_t at_least, std::size_t preferred) noexcept {
        this->lent_at = this->length;
        const auto needed = std::max(at_least, std::size_t { 1 });
        if (this->capacity() - this->length < needed && !this->reserve(this->length + std::max(needed, preferred))) return {};
        return { this->data() + this->length, this->capacity() - this->length };
    }

    void keep(std::size_t used) noexcept { this->length = this->lent_at + used; }

    /** The document, as the bytes it was written into. */
    [[nodiscard]] nb::bytes take() {
        if (this->bytes == nullptr || _PyBytes_Resize(&this->bytes, static_cast<Py_ssize_t>(this->length)) < 0) propagate();
        return nb::steal<nb::bytes>(std::exchange(this->bytes, nullptr));
    }
};

template<prefer Preference>
nb::bytes encode_as(nb::handle value, const encode_options &options) {
    bytes_sink sink;
    {
        basic_writer<Preference> out { sink };
        encoder<Preference> { out, options }.value(value.ptr());
        if (const auto finished = out.finish(); !finished) {
            if (PyErr_Occurred()) propagate();
            raise_encode_error(serpent::message(finished.error().code()));
        }
    }
    return sink.take();
}

} // namespace

nb::bytes encode(nb::handle value, const encode_options &options) {
    return options.speed ? encode_as<prefer::speed>(value, options) : encode_as<prefer::size>(value, options);
}

} // namespace bjdata_serpent
