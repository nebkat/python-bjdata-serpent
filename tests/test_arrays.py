"""Packed and N-D arrays, decoded as lists; numpy has its own tests."""

import array
import struct

import pytest

import bjdata_serpent as bjd


@pytest.mark.parametrize(
    "marker, code, values",
    [
        (b"U", "B", [0, 1, 255]),
        (b"i", "b", [-128, 0, 127]),
        (b"u", "H", [0, 65535]),
        (b"I", "h", [-32768, 32767]),
        (b"m", "I", [0, 2**32 - 1]),
        (b"l", "i", [-(2**31), 2**31 - 1]),
        (b"M", "Q", [0, 2**64 - 1]),
        (b"L", "q", [-(2**63), 2**63 - 1]),
        (b"h", "e", [1.0, -2.0, 0.5]),
        (b"d", "f", [1.5, -0.25]),
        (b"D", "d", [0.1, 1e300]),
    ],
)
def test_packed_arrays_decode_to_lists(marker, code, values):
    document = b"[$" + marker + b"#U" + bytes([len(values)]) + struct.pack("<%d%s" % (len(values), code), *values)
    assert bjd.decode(document) == values


def test_nd_arrays_decode_to_nested_lists():
    assert bjd.decode(b"[$U#[U\x02U\x03]\x01\x02\x03\x04\x05\x06") == [[1, 2, 3], [4, 5, 6]]
    assert bjd.decode(b"[$I#[U\x02U\x01U\x02]" + struct.pack("<4h", 1, 2, 3, 4)) == [[[1, 2]], [[3, 4]]]


def test_column_major_nd_arrays_are_laid_out_leftmost_fastest():
    assert bjd.decode(b"[$U#[[U\x02U\x03]]\x01\x02\x03\x04\x05\x06") == [[1, 3, 5], [2, 4, 6]]


def test_empty_packed_arrays():
    assert bjd.decode(b"[$l#U\x00") == []


def test_counted_arrays_of_bytes_are_binary_and_of_uint8_numbers():
    assert bjd.decode(b"[$B#U\x02\x01\x02") == b"\x01\x02"
    assert bjd.decode(b"[$U#U\x02\x01\x02") == [1, 2]


def test_typed_arrays_of_characters():
    assert bjd.decode(b"[$C#U\x02ab") == ["a", "b"]


def test_array_module_arrays_encode_as_packed():
    assert bjd.encode(array.array("i", [1, 2])) == b"[$l#U\x02" + struct.pack("<2i", 1, 2)
    assert bjd.encode(array.array("d", [0.5])) == b"[$D#U\x01" + struct.pack("<d", 0.5)
    assert bjd.decode(bjd.encode(array.array("h", [-1, 300]))) == [-1, 300]


def test_byte_buffers_encode_as_binary():
    assert bjd.encode(array.array("B", [1, 2])) == b"[$B#U\x02\x01\x02"


def test_memoryviews_of_numbers_encode_as_packed():
    view = memoryview(array.array("H", [1, 2, 3]))
    assert bjd.decode(bjd.encode(view)) == [1, 2, 3]
    assert bjd.decode(bjd.encode(view[::2])) == [1, 3]


def test_multi_dimensional_memoryviews_keep_their_shape():
    view = memoryview(struct.pack("<6i", *range(6))).cast("B").cast("i", (2, 3))
    encoded = bjd.encode(view)
    assert encoded.startswith(b"[$l#[U\x02U\x03]")
    assert bjd.decode(encoded) == [[0, 1, 2], [3, 4, 5]]
