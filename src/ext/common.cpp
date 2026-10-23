#include "common.hpp"

#include <string>

namespace bjdata_serpent {

std::string_view code_name(serpent::errc code) noexcept {
    using serpent::errc;
    switch (code) {
    case errc::ok: return "ok";
    case errc::unexpected_end: return "unexpected_end";
    case errc::trailing_data: return "trailing_data";
    case errc::invalid_marker: return "invalid_marker";
    case errc::unexpected_marker: return "unexpected_marker";
    case errc::extension_unsupported: return "extension_unsupported";
    case errc::depth_exceeded: return "depth_exceeded";
    case errc::invalid_length: return "invalid_length";
    case errc::negative_length: return "negative_length";
    case errc::invalid_strong_type: return "invalid_strong_type";
    case errc::missing_count: return "missing_count";
    case errc::unterminated_container: return "unterminated_container";
    case errc::invalid_dimensions: return "invalid_dimensions";
    case errc::dimension_overflow: return "dimension_overflow";
    case errc::object_dimension_count: return "object_dimension_count";
    case errc::type_mismatch: return "type_mismatch";
    case errc::out_of_range: return "out_of_range";
    case errc::missing_key: return "missing_key";
    case errc::unexpected_character: return "unexpected_character";
    case errc::invalid_number: return "invalid_number";
    case errc::invalid_string: return "invalid_string";
    case errc::invalid_escape: return "invalid_escape";
    case errc::sink_failed: return "sink_failed";
    case errc::unbalanced_container: return "unbalanced_container";
    case errc::key_outside_object: return "key_outside_object";
    }
    return "unknown";
}

namespace {

[[noreturn]] void raise(const char *type, nb::handle arguments) {
    const auto errors = owned(PyImport_ImportModule("bjdata_serpent._errors"));
    const auto cls = owned(PyObject_GetAttrString(errors.ptr(), type));
    const auto instance = owned(PyObject_Call(cls.ptr(), arguments.ptr(), nullptr));
    PyErr_SetObject(cls.ptr(), instance.ptr());
    propagate();
}

[[noreturn]] void raise_decode(std::string_view message, std::size_t offset, nb::handle code) {
    const auto text = std::string { message } + " at byte " + std::to_string(offset);
    raise("DecodeError", nb::make_tuple(nb::str(text.data(), text.size()), offset, code));
}

} // namespace

void raise_decode_error(serpent::errc code, std::size_t offset) {
    const auto name = code_name(code);
    raise_decode(serpent::message(code), offset, nb::str(name.data(), name.size()));
}

void raise_decode_error(std::string_view message, std::size_t offset) {
    raise_decode(message, offset, nb::none());
}

void raise_encode_error(std::string_view message) {
    raise("EncodeError", nb::make_tuple(nb::str(message.data(), message.size())));
}

nb::handle numpy() {
    // Held for the life of the interpreter, as sys.modules holds it anyway.
    static PyObject *module = nullptr;
    if (module == nullptr) {
        module = PyImport_ImportModule("numpy");
        if (module == nullptr) propagate();
    }
    return module;
}

PyObject *imported(const char *name) noexcept {
    PyObject *const modules = PyImport_GetModuleDict();
    return PyDict_GetItemString(modules, name);
}

nb::handle decimal_type() {
    static PyObject *type = nullptr;
    if (type == nullptr) {
        const auto module = owned(PyImport_ImportModule("decimal"));
        type = PyObject_GetAttrString(module.ptr(), "Decimal");
        if (type == nullptr) propagate();
    }
    return type;
}

} // namespace bjdata_serpent
