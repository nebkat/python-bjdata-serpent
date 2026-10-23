"""numpy arrays and scalars, in and out."""

import pytest

import bjdata_serpent as bjd

np = pytest.importorskip("numpy")

NUMBERS = ["u1", "i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "<f2", "<f4", "<f8"]


@pytest.mark.parametrize("dtype", NUMBERS)
def test_arrays_round_trip_with_their_dtype(dtype):
    values = np.arange(12, dtype=dtype)
    decoded = bjd.decode(bjd.encode(values), numpy=True)
    assert decoded.dtype == np.dtype(dtype)
    assert np.array_equal(decoded, values)
    assert decoded.flags.writeable


@pytest.mark.parametrize("dtype", NUMBERS)
def test_arrays_decode_to_lists_unless_asked(dtype):
    assert bjd.decode(bjd.encode(np.arange(3, dtype=dtype))) == [0, 1, 2]


def test_uint8_arrays_are_numbers_not_bytes():
    encoded = bjd.encode(np.array([1, 2], dtype="u1"))
    assert encoded == b"[$U#U\x02\x01\x02"


def test_nd_arrays_keep_their_shape():
    values = np.arange(24, dtype="<i4").reshape(2, 3, 4)
    encoded = bjd.encode(values)
    assert encoded.startswith(b"[$l#[U\x02U\x03U\x04]")
    assert np.array_equal(bjd.decode(encoded, numpy=True), values)
    assert bjd.decode(encoded) == values.tolist()


def test_column_major_documents_decode_to_fortran_ordered_arrays():
    decoded = bjd.decode(b"[$U#[[U\x02U\x03]]\x01\x02\x03\x04\x05\x06", numpy=True)
    assert decoded.tolist() == [[1, 3, 5], [2, 4, 6]]
    assert decoded.flags.f_contiguous


@pytest.mark.parametrize(
    "values",
    [
        np.arange(20, dtype="<i4").reshape(4, 5)[:, ::2],
        np.arange(20, dtype="<f8").reshape(4, 5).T,
        np.arange(6, dtype=">i4"),
        np.arange(6, dtype=">f8").reshape(2, 3),
    ],
    ids=["strided", "transposed", "big-endian", "big-endian 2-D"],
)
def test_arrays_not_in_order_are_written_in_order(values):
    decoded = bjd.decode(bjd.encode(values), numpy=True)
    assert np.array_equal(decoded, values)
    assert decoded.dtype == values.dtype.newbyteorder("<")


def test_empty_arrays():
    assert bjd.decode(bjd.encode(np.zeros(0, dtype="<f4")), numpy=True).shape == (0,)
    assert bjd.decode(bjd.encode(np.zeros((2, 0), dtype="<i2")), numpy=True).shape == (2, 0)


def test_arrays_of_other_kinds_become_lists():
    assert bjd.decode(bjd.encode(np.array([True, False]))) == [True, False]
    assert bjd.decode(bjd.encode(np.array(["ab", "c"]))) == ["ab", "c"]
    assert bjd.decode(bjd.encode(np.array([[True], [False]]))) == [[True], [False]]
    assert bjd.decode(bjd.encode(np.array([1, "a"], dtype=object))) == [1, "a"]


@pytest.mark.parametrize(
    "scalar, marker",
    [
        (np.int8(-3), b"i"),
        (np.uint16(7), b"u"),
        (np.int32(7), b"l"),
        (np.int64(7), b"L"),
        (np.float16(1.5), b"h"),
        (np.float32(1.5), b"d"),
        (np.float64(1.5), b"D"),
    ],
)
def test_scalars_keep_their_width_for_speed(scalar, marker):
    encoded = bjd.encode(scalar, prefer="speed")
    assert encoded[:1] == marker
    assert bjd.decode(encoded) == scalar


def test_scalars_narrow_for_size():
    assert bjd.encode(np.int64(7)) == b"U\x07"
    assert bjd.encode(np.float32(1.5)) == b"h\x00\x3e"
    assert bjd.encode(np.bool_(True)) == b"T"


def test_float16_arrays_decode_to_lists():
    assert bjd.decode(bjd.encode(np.array([0.5, -1.0], dtype="<f2"))) == [0.5, -1.0]


def test_arrays_inside_containers():
    document = {"samples": np.arange(3, dtype="<i2"), "name": "adc"}
    decoded = bjd.decode(bjd.encode(document), numpy=True)
    assert decoded["name"] == "adc"
    assert decoded["samples"].dtype == np.dtype("<i2")


def test_more_dimensions_than_bjdata_allows():
    with pytest.raises(bjd.EncodeError, match="more than 8 dimensions"):
        bjd.encode(np.zeros((1,) * 9, dtype="u1"))
