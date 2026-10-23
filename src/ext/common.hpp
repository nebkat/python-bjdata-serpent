#pragma once

// What the decoder and the encoder share: raising the module's errors, and the modules that are
// imported only when a value needs them.

#include <nanobind/nanobind.h>
#include <serpent/error.hpp>

#include <cstddef>
#include <string_view>

namespace bjdata_serpent {

namespace nb = nanobind;

/** Unwinds to nanobind, which raises the Python exception already set. */
[[noreturn]] inline void propagate() { throw nb::python_error(); }

/** Takes a new reference, or propagates the exception the call that failed to return one set. */
[[nodiscard]] inline nb::object owned(PyObject *object) {
    if (object == nullptr) propagate();
    return nb::steal(object);
}

/** An object's buffer, held for as long as this is. */
class buffer_view {
    Py_buffer held {};

public:
    buffer_view(PyObject *object, int flags) {
        if (PyObject_GetBuffer(object, &this->held, flags) < 0) propagate();
    }
    buffer_view(const buffer_view &) = delete;
    buffer_view &operator=(const buffer_view &) = delete;
    ~buffer_view() { PyBuffer_Release(&this->held); }

    [[nodiscard]] const Py_buffer &operator*() const noexcept { return this->held; }
    [[nodiscard]] const Py_buffer *operator->() const noexcept { return &this->held; }
};

/** serpent's name for an error code - the `code` a DecodeError carries. */
[[nodiscard]] std::string_view code_name(serpent::errc code) noexcept;

/** Raises DecodeError, saying what went wrong and at which byte. */
[[noreturn]] void raise_decode_error(serpent::errc code, std::size_t offset);
[[noreturn]] void raise_decode_error(std::string_view message, std::size_t offset);

/** Raises EncodeError. */
[[noreturn]] void raise_encode_error(std::string_view message);

/** numpy, imported the first time an array asks for it; ImportError when it is not installed. */
[[nodiscard]] nb::handle numpy();

/**
 * A module that has been imported already, or null. An object cannot be an instance of a type
 * from a module nobody has imported, so asking this costs nothing and imports nothing.
 */
[[nodiscard]] PyObject *imported(const char *name) noexcept;

/** decimal.Decimal. */
[[nodiscard]] nb::handle decimal_type();

} // namespace bjdata_serpent
