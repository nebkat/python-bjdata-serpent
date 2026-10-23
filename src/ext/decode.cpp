#include "decode.hpp"

#include <serpent/bjdata/marker.hpp>
#include <serpent/bjdata/reader.hpp>
#include <serpent/bjdata/soa.hpp>
#include <serpent/limits.hpp>

#include <nonstd/unaligned_ptr.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bjdata_serpent {

namespace {

using namespace nb::literals;
using serpent::errc;
using serpent::bjdata::marker;
using serpent::bjdata::reader;
namespace soa = serpent::bjdata::soa;

/** The numpy dtype of a fixed-width number marker, or null for any other marker. */
[[nodiscard]] const char *number_dtype(marker element) noexcept {
    switch (element) {
    case marker::uint8:
    case marker::byte: return "u1";
    case marker::int8: return "i1";
    case marker::uint16: return "<u2";
    case marker::int16: return "<i2";
    case marker::uint32: return "<u4";
    case marker::int32: return "<i4";
    case marker::uint64: return "<u8";
    case marker::int64: return "<i8";
    case marker::float16: return "<f2";
    case marker::float32: return "<f4";
    case marker::float64: return "<f8";
    default: return nullptr;
    }
}

/** Strides, in elements, of an array of these extents laid out one way round or the other. */
[[nodiscard]] std::vector<std::size_t> strides_of(std::span<const std::uint32_t> extents, bool column_major) {
    std::vector<std::size_t> strides(extents.size());
    std::size_t step = 1;
    if (column_major) {
        for (std::size_t index = 0; index < extents.size(); ++index) {
            strides[index] = step;
            step *= extents[index];
        }
    } else {
        for (std::size_t index = extents.size(); index-- > 0;) {
            strides[index] = step;
            step *= extents[index];
        }
    }
    return strides;
}

/** The elements of `flat` - in the order the document holds them - as lists nested to `extents`. */
[[nodiscard]] nb::object nest(PyObject *flat, std::span<const std::uint32_t> extents, std::span<const std::size_t> strides,
        std::size_t base) {
    auto list = owned(PyList_New(static_cast<Py_ssize_t>(extents[0])));
    for (std::size_t index = 0; index < extents[0]; ++index) {
        const auto at = base + index * strides[0];
        PyObject *item = nullptr;
        if (extents.size() == 1) {
            item = PyList_GET_ITEM(flat, static_cast<Py_ssize_t>(at));
            Py_INCREF(item);
        } else {
            item = nest(flat, extents.subspan(1), strides.subspan(1), at).release().ptr();
        }
        PyList_SET_ITEM(list.ptr(), static_cast<Py_ssize_t>(index), item);
    }
    return list;
}

[[nodiscard]] nb::object nest(const nb::object &flat, const serpent::bjdata::detail::header &info) {
    const std::span<const std::uint32_t> extents { info.extents, info.rank };
    // The document is checked only once it has been walked, so the elements have to be counted.
    std::uint64_t total = 1;
    for (const auto extent : extents) total *= extent;
    if (static_cast<std::uint64_t>(PyList_GET_SIZE(flat.ptr())) != total) raise_decode_error(errc::invalid_dimensions, 0);
    const auto strides = strides_of(extents, info.column_major);
    return nest(flat.ptr(), extents, strides, 0);
}

[[nodiscard]] nb::tuple shape_of(const serpent::bjdata::detail::header &info) {
    if (info.rank == 0) return nb::make_tuple(info.count);
    auto shape = owned(PyTuple_New(static_cast<Py_ssize_t>(info.rank)));
    for (std::size_t index = 0; index < info.rank; ++index)
        PyTuple_SET_ITEM(shape.ptr(), static_cast<Py_ssize_t>(index), PyLong_FromUnsignedLong(info.extents[index]));
    return nb::borrow<nb::tuple>(shape);
}

/**
 * Keys made once and handed out again, for the objects of an array that all have the same ones.
 *
 * Direct-mapped and fixed in size, so a document with a handful of keys - most of them - costs
 * no allocation for this at all; a collision only means a key is made again.
 */
class key_cache {
    static constexpr std::size_t slots = 64;
    static constexpr std::size_t longest = 64;
    // A name is only looked at where its slot has a key made, so only those are zeroed: this is
    // made for every document decoded, and most hold a handful of keys.
    struct held_name {
        const char *data;
        std::size_t size;
    };
    held_name names[slots];
    PyObject *made[slots] {};

    [[nodiscard]] static std::size_t slot_of(std::string_view name) noexcept {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (const char byte : name) hash = (hash ^ static_cast<unsigned char>(byte)) * 0x100000001b3ull;
        return static_cast<std::size_t>(hash ^ (hash >> 32)) & (slots - 1);
    }

public:
    key_cache() = default;
    key_cache(const key_cache &) = delete;
    key_cache &operator=(const key_cache &) = delete;
    ~key_cache() {
        for (PyObject *const key : this->made) Py_XDECREF(key);
    }

    template<typename Make>
    [[nodiscard]] nb::object get(std::string_view name, Make make) {
        if (name.size() > longest) return make();
        const auto slot = slot_of(name);
        if (this->made[slot] != nullptr && std::string_view { this->names[slot].data, this->names[slot].size } == name)
            return nb::borrow(this->made[slot]);
        auto key = make();
        Py_XDECREF(this->made[slot]);
        this->made[slot] = key.inc_ref().ptr();
        this->names[slot] = { name.data(), name.size() };
        return key;
    }
};

class decoder {
    const decode_options &options;
    key_cache keys;
    int depth = 0;

    /**
     * Counts a container for as long as it is being read. The document is validated only once it
     * has been read, so the reading has to stop at the depth validation would, rather than recurse
     * as deep as a hostile document nests.
     */
    class level {
        decoder &self;

    public:
        level(decoder &self, const reader &source) : self(self) {
            if (++self.depth > serpent::max_depth) {
                --self.depth;
                raise_decode_error(errc::depth_exceeded, source.offset());
            }
        }
        level(const level &) = delete;
        level &operator=(const level &) = delete;
        ~level() { --this->self.depth; }
    };

    template<typename T>
    [[nodiscard]] static T present(const std::optional<T> &value, const reader &source) {
        if (!value) raise_decode_error(errc::type_mismatch, source.offset());
        return *value;
    }

    [[nodiscard]] static nb::object text(std::string_view bytes, std::size_t offset) {
        PyObject *const decoded = PyUnicode_FromStringAndSize(bytes.data(), static_cast<Py_ssize_t>(bytes.size()));
        if (decoded == nullptr) {
            PyErr_Clear();
            raise_decode_error("text that is not UTF-8", offset);
        }
        return nb::steal(decoded);
    }

    [[nodiscard]] nb::object key(std::string_view name, std::size_t offset) {
        return this->keys.get(name, [&] { return text(name, offset); });
    }

    [[nodiscard]] nb::object list_of(const reader &source) { return this->list_of(source, source.container_header()); }

    /** The elements of an array whose header has been read, into a list sized once when it is counted. */
    [[nodiscard]] nb::object list_of(const reader &source, const serpent::bjdata::detail::header &info) {
        serpent::bjdata::array_iterator element { source, info };
        // Sized once only when the bytes are there for that many: each element takes one at least.
        const auto *const limit = source.buffer().data() + source.buffer().size();
        if (!info.unbounded && !source.place_in_table().in_table() && info.body != nullptr
                && info.count <= static_cast<std::uint64_t>(limit - info.body)) {
            auto list = owned(PyList_New(static_cast<Py_ssize_t>(info.count)));
            Py_ssize_t index = 0;
            for (; element != std::default_sentinel && index < static_cast<Py_ssize_t>(info.count); ++element)
                PyList_SET_ITEM(list.ptr(), index++, this->value(*element).release().ptr());
            if (index != static_cast<Py_ssize_t>(info.count)) raise_decode_error(errc::unexpected_end, source.offset());
            return list;
        }
        auto list = owned(PyList_New(0));
        for (; element != std::default_sentinel; ++element) {
            const auto item = this->value(*element);
            if (PyList_Append(list.ptr(), item.ptr()) < 0) propagate();
        }
        return list;
    }

    template<typename T>
    [[nodiscard]] static nb::object numbers(const reader &source, std::uint64_t count) {
        const auto values = present(source.as<nonstd::unaligned_little_span<const T>>(), source);
        auto list = owned(PyList_New(static_cast<Py_ssize_t>(count)));
        Py_ssize_t index = 0;
        for (const T item : values) {
            PyObject *number = nullptr;
            if constexpr (std::floating_point<T>) number = PyFloat_FromDouble(item);
            else if constexpr (std::is_signed_v<T>) number = PyLong_FromLongLong(item);
            else number = PyLong_FromUnsignedLongLong(item);
            if (number == nullptr) propagate();
            PyList_SET_ITEM(list.ptr(), index++, number);
        }
        return list;
    }

    /** A typed array of numbers: a numpy array of its type and shape, or (nested) lists. */
    [[nodiscard]] nb::object packed(const reader &source, const serpent::bjdata::detail::header &info) {
        if (this->options.numpy) {
            // Checked before numpy is asked for the room: the count is the document's say-so.
            const auto width = serpent::bjdata::payload_width(info.element);
            const auto *const limit = source.buffer().data() + source.buffer().size();
            if (info.body == nullptr || info.count > static_cast<std::uint64_t>(limit - info.body) / width)
                raise_decode_error(errc::unexpected_end, source.offset());
            auto array = numpy().attr("empty")(shape_of(info), "dtype"_a = number_dtype(info.element),
                    "order"_a = info.column_major ? "F" : "C");
            const buffer_view into { array.ptr(), PyBUF_WRITABLE | PyBUF_ANY_CONTIGUOUS };
            const auto bytes = info.count * width;
            if (bytes != 0) std::memcpy(into->buf, info.body, static_cast<std::size_t>(bytes));
            return array;
        }

        nb::object flat;
        switch (info.element) {
        case marker::uint8: flat = numbers<std::uint8_t>(source, info.count); break;
        case marker::int8: flat = numbers<std::int8_t>(source, info.count); break;
        case marker::uint16: flat = numbers<std::uint16_t>(source, info.count); break;
        case marker::int16: flat = numbers<std::int16_t>(source, info.count); break;
        case marker::uint32: flat = numbers<std::uint32_t>(source, info.count); break;
        case marker::int32: flat = numbers<std::int32_t>(source, info.count); break;
        case marker::uint64: flat = numbers<std::uint64_t>(source, info.count); break;
        case marker::int64: flat = numbers<std::int64_t>(source, info.count); break;
        case marker::float32: flat = numbers<float>(source, info.count); break;
        case marker::float64: flat = numbers<double>(source, info.count); break;
        default: flat = this->list_of(source); break; // float16 and bytes counted by dimensions
        }
        return info.rank > 1 ? nest(flat, info) : flat;
    }

    [[nodiscard]] nb::object array(const reader &source) {
        // A fixed run of elements in a record of a table: its shape is the schema's.
        if (source.place_in_table().in_table()) return this->list_of(source);

        const auto info = source.container_header();
        if (info.structured()) return this->table(source, info);
        if (info.typed()) {
            if (info.element == marker::byte && info.rank == 0) {
                const auto bytes = present(source.as<std::span<const std::byte>>(), source);
                return owned(PyBytes_FromStringAndSize(reinterpret_cast<const char *>(bytes.data()),
                        static_cast<Py_ssize_t>(bytes.size())));
            }
            if (number_dtype(info.element) != nullptr) return this->packed(source, info);
        }
        auto flat = this->list_of(source, info);
        return info.rank > 1 ? nest(flat, info) : flat;
    }

    [[nodiscard]] nb::object object(const reader &source) {
        if (this->options.object_pairs_hook.is_valid()) {
            auto pairs = owned(PyList_New(0));
            for (const auto &entry : source.items()) {
                const auto pair = nb::make_tuple(this->key(entry.key, entry.value.offset()), this->value(entry.value));
                if (PyList_Append(pairs.ptr(), pair.ptr()) < 0) propagate();
            }
            return this->options.object_pairs_hook(pairs);
        }
        auto dict = owned(PyDict_New());
        for (const auto &entry : source.items()) {
            const auto name = this->key(entry.key, entry.value.offset());
            const auto item = this->value(entry.value);
            if (PyDict_SetItem(dict.ptr(), name.ptr(), item.ptr()) < 0) propagate();
        }
        if (this->options.object_hook.is_valid()) return this->options.object_hook(dict);
        return dict;
    }

    // ---------------- tables ----------------

    /** The numpy dtype of the fields `first` up to `last` of a schema, as numpy takes a list of them. */
    [[nodiscard]] nb::object record_dtype(const soa::schema &schema, std::uint32_t first, std::uint32_t last, bool named) {
        auto fields = owned(PyList_New(0));
        std::uint32_t position = 0;
        for (auto index = first; index < last; index = schema.fields[index].next, ++position) {
            const auto &field = schema.fields[index];
            const auto name = named ? nb::borrow(this->key(field.name, 0)) : nb::object(nb::str(("f" + std::to_string(position)).c_str()));
            const auto entry = this->field_dtype(schema, field, name);
            if (PyList_Append(fields.ptr(), entry.ptr()) < 0) propagate();
        }
        return fields;
    }

    /** One field of a numpy dtype: (name, type) or, for a run of one type, (name, type, (count,)). */
    [[nodiscard]] nb::tuple field_dtype(const soa::schema &schema, const soa::field &field, const nb::object &name) {
        switch (field.kind) {
        case soa::field_kind::scalar:
            if (field.type == marker::character) return nb::make_tuple(name, "U1");
            return nb::make_tuple(name, number_dtype(field.type));
        case soa::field_kind::boolean: return nb::make_tuple(name, "?");
        case soa::field_kind::text: return nb::make_tuple(name, "U" + std::to_string(field.width));
        case soa::field_kind::null:
        case soa::field_kind::dictionary:
        case soa::field_kind::offsets: return nb::make_tuple(name, "O");
        case soa::field_kind::record: return nb::make_tuple(name, this->record_dtype(schema, field.children, field.next, true));
        case soa::field_kind::array: {
            // A run of one number type is a subarray; anything else, a record of unnamed fields.
            const auto &first = schema.fields[field.children];
            bool uniform = first.kind == soa::field_kind::scalar && number_dtype(first.type) != nullptr;
            std::uint32_t count = 0;
            for (auto index = field.children; index < field.next; index = schema.fields[index].next, ++count) {
                const auto &element = schema.fields[index];
                if (element.kind != first.kind || element.type != first.type) uniform = false;
            }
            if (uniform) return nb::make_tuple(name, number_dtype(first.type), nb::make_tuple(count));
            return nb::make_tuple(name, this->record_dtype(schema, field.children, field.next, false));
        }
        }
        raise_decode_error(errc::invalid_strong_type, 0);
    }

    /**
     * The fields `first` up to `last` of one record, whose bytes start at `at`: a dict, or for
     * numpy the tuple it fills a row from. A nested record is the same again from its own bytes;
     * a run is a list, or a tuple.
     *
     * Walked from the placed table rather than through reader handles, which place the table
     * again - reading its schema - for every record they step into.
     */
    [[nodiscard]] nb::object fields_of(std::span<const std::byte> document, const soa::table &layout, const std::byte *container,
            std::uint32_t first, std::uint32_t last, std::uint64_t record, const std::byte *at, bool top, bool named) {
        const auto &schema = layout.fields();
        const bool as_tuple = this->options.numpy || !named;
        auto values = owned(as_tuple ? PyList_New(0) : PyDict_New());
        for (auto index = first; index < last; index = schema.fields[index].next) {
            const auto &field = schema.fields[index];
            const std::byte *const here = top ? layout.at(field, record) : at + field.offset;
            nb::object item;
            if (field.kind == soa::field_kind::record) {
                item = this->fields_of(document, layout, container, field.children, field.next, record, here, false, true);
            } else if (field.kind == soa::field_kind::array) {
                item = this->fields_of(document, layout, container, field.children, field.next, record, here, false, false);
            } else {
                reader handle { marker::invalid, document, nullptr };
                serpent::bjdata::place_record(handle, container, record);
                serpent::bjdata::place_field(handle, layout, field, here);
                item = this->value(handle);
            }
            const int added = as_tuple ? PyList_Append(values.ptr(), item.ptr())
                                       : PyDict_SetItem(values.ptr(), this->key(field.name, 0).ptr(), item.ptr());
            if (added < 0) propagate();
        }
        if (!as_tuple) {
            if (this->options.object_pairs_hook.is_valid()) return this->options.object_pairs_hook(owned(PyDict_Items(values.ptr())));
            if (this->options.object_hook.is_valid()) return this->options.object_hook(values);
            return values;
        }
        // A run outside numpy is a list, as any array is; what numpy fills a row from is a tuple.
        return this->options.numpy ? owned(PyList_AsTuple(values.ptr())) : values;
    }

    /** A Structure-of-Arrays table: a numpy structured array, or a list of dicts. */
    [[nodiscard]] nb::object table(const reader &source, const serpent::bjdata::detail::header &info) {
        soa::table layout;
        if (!layout.place(source.buffer(), info, source.type_marker() == marker::object_begin))
            raise_decode_error(errc::invalid_dimensions, source.offset());
        const auto &schema = layout.fields();
        const std::byte *const container = source.data() - 1;
        // Records of nothing but null fields take no bytes, so their count is bounded by nothing
        // in the document: more of them than it has bytes is refused rather than made.
        if (schema.record_bytes == 0 && layout.size() > source.buffer().size())
            raise_decode_error(errc::dimension_overflow, source.offset());

        auto rows = owned(PyList_New(static_cast<Py_ssize_t>(layout.size())));
        for (std::uint64_t record = 0; record < layout.size(); ++record) {
            auto row = this->fields_of(source.buffer(), layout, container, 0, schema.count(), record, nullptr, true, true);
            PyList_SET_ITEM(rows.ptr(), static_cast<Py_ssize_t>(record), row.release().ptr());
        }
        if (!this->options.numpy) return info.rank > 1 ? nest(rows, info) : rows;

        auto array = numpy().attr("array")(rows, "dtype"_a = this->record_dtype(schema, 0, schema.count(), true));
        if (info.rank > 1) array = array.attr("reshape")(shape_of(info));
        return array;
    }

public:
    explicit decoder(const decode_options &options) noexcept : options(options) {}

    [[nodiscard]] nb::object value(const reader &source) {
        switch (source.type_marker()) {
        case marker::null: return nb::none();
        case marker::boolean_true: return nb::borrow(Py_True);
        case marker::boolean_false: return nb::borrow(Py_False);
        case marker::uint64: return owned(PyLong_FromUnsignedLongLong(present(source.as<std::uint64_t>(), source)));
        case marker::uint8:
        case marker::int8:
        case marker::uint16:
        case marker::int16:
        case marker::uint32:
        case marker::int32:
        case marker::int64:
        case marker::byte: return owned(PyLong_FromLongLong(present(source.as<std::int64_t>(), source)));
        case marker::float16:
        case marker::float32:
        case marker::float64: return owned(PyFloat_FromDouble(present(source.as<double>(), source)));
        case marker::character:
        case marker::string: return text(present(source.as<std::string_view>(), source), source.offset());
        case marker::high_precision: {
            // An integer too wide for any integer marker comes back as the int it was; anything
            // else as a Decimal, which holds every digit.
            const auto bytes = present(source.as<std::string_view>(), source);
            const auto sign = !bytes.empty() && (bytes[0] == '-' || bytes[0] == '+') ? 1u : 0u;
            if (bytes.size() > sign && std::ranges::all_of(bytes.substr(sign), [](char c) { return c >= '0' && c <= '9'; })) {
                const std::string terminated { bytes };
                return owned(PyLong_FromString(terminated.c_str(), nullptr, 10));
            }
            const auto digits = text(bytes, source.offset());
            PyObject *const number = PyObject_CallFunctionObjArgs(decimal_type().ptr(), digits.ptr(), nullptr);
            if (number == nullptr) {
                PyErr_Clear();
                raise_decode_error("a high-precision number that is not one", source.offset());
            }
            return nb::steal(number);
        }
        case marker::array_begin: {
            const level nested { *this, source };
            return this->array(source);
        }
        case marker::object_begin: {
            const level nested { *this, source };
            // A table laid out column by column opens with `{` but is a sequence of records.
            if (source.is_array()) return this->table(source, source.container_header());
            return this->object(source);
        }
        default: raise_decode_error(errc::unexpected_marker, source.offset());
        }
    }
};

} // namespace

void validate(std::span<const std::byte> document) {
    if (const auto checked = serpent::bjdata::validate(document); !checked)
        raise_decode_error(checked.error().code(), checked.error().offset());
}

namespace {

/** The part of the document that has to be well formed: all of it, or the value it starts with. */
[[nodiscard]] std::optional<std::span<const std::byte>> checked_part(std::span<const std::byte> document, bool whole) {
    if (whole) return document;
    const auto root = reader::over(document);
    const auto *const end = root ? root.value_end() : nullptr;
    if (end == nullptr) return std::nullopt;
    return document.first(static_cast<std::size_t>(end - document.data()));
}

/** Raises DecodeError for what is wrong with the part of the document that has to be well formed. */
[[noreturn]] void raise_malformed(std::span<const std::byte> document, bool whole) {
    // A leading value that does not end is wrong somewhere before where the document would.
    validate(checked_part(document, whole).value_or(document));
    raise_decode_error(errc::unexpected_end, document.size()); // validates, yet did not read: not expected
}

/**
 * The value the document starts with, and where it ends; with `whole`, only a value that is the
 * entire document.
 *
 * Read first and checked after, in the one walk: the walk leaves a note of where each container
 * it finished ended, and only a value that parses all the way through has an end. A document
 * that does not is then validated, for what is wrong with it and where - which is also the
 * answer when something broken trips the walk before the check is reached.
 */
std::pair<nb::object, std::size_t> read_value(std::span<const std::byte> document, const decode_options &options, bool whole) {
    if (const auto root = reader::over(document)) {
        try {
            decoder values { options };
            auto value = values.value(root);
            const auto *const end = root.value_end();
            if (end != nullptr && (!whole || end == document.data() + document.size()))
                return { std::move(value), static_cast<std::size_t>(end - document.data()) };
        } catch (nb::python_error &) {
            // A hook raised, or text was not UTF-8, in a document that is otherwise fine: that is
            // the error. Otherwise the document is broken, and validation says how.
            const auto part = checked_part(document, whole);
            if (part && serpent::bjdata::validate(*part)) throw;
        }
    }
    raise_malformed(document, whole);
}

} // namespace

nb::object decode(std::span<const std::byte> document, const decode_options &options) {
    return read_value(document, options, true).first;
}

nb::tuple decode_prefix(std::span<const std::byte> document, const decode_options &options) {
    auto [value, end] = read_value(document, options, false);
    return nb::make_tuple(value, end);
}

} // namespace bjdata_serpent
