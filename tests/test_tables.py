"""Structure-of-Arrays tables, from lists of dicts and from structured numpy arrays."""

import pytest

import bjdata_serpent as bjd

RECORDS = [
    {"id": 1, "value": 0.5, "ok": True, "name": "a", "pos": {"x": 1, "y": -1}, "v": [1.0, 2.0], "none": None},
    {"id": 2, "value": 1.5, "ok": False, "name": "bcd", "pos": {"x": 2, "y": -2}, "v": [3.0, 4.0], "none": None},
    {"id": 300, "value": 2.5, "ok": True, "name": "", "pos": {"x": 3, "y": -3}, "v": [5.0, 6.5], "none": None},
]


def is_table(encoded):
    return encoded[:3] in (b"[${", b"{${")


@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("prefer", ["size", "speed"])
def test_lists_of_like_dicts_are_tables(order, prefer):
    encoded = bjd.encode(RECORDS, tables=True, table_order=order, prefer=prefer)
    assert is_table(encoded)
    assert encoded[:1] == (b"[" if order == "row" else b"{")
    assert bjd.decode(encoded) == RECORDS
    assert len(encoded) < len(bjd.encode(RECORDS, prefer=prefer))


def test_tables_only_when_asked():
    assert not is_table(bjd.encode(RECORDS))


@pytest.mark.parametrize(
    "records",
    [
        [{"a": 1}],
        [{"a": 1}, {"b": 1}],
        [{"a": 1}, {"a": 1, "b": 2}],
        [{"a": 1}, {"a": "x"}],
        [{"a": 1}, {"a": None}],
        [{"a": [1]}, {"a": [1, 2]}],
        [{"a": []}, {"a": []}],
        [{}, {}],
        [{"a": 1}, 5],
        [{"a": 2**64}, {"a": 1}],
        [{"a": -1}, {"a": 2**63}],
        [{"a": 2**60}, {"a": 0.5}],
        [{1: 1}, {1: 1}],
    ],
)
def test_anything_a_schema_cannot_say_is_an_array_of_objects(records):
    try:
        encoded = bjd.encode(records, tables=True)
    except TypeError:
        return  # keys that are not str are refused either way
    assert not is_table(encoded)
    assert bjd.decode(encoded) == records


def test_text_columns_are_characters_dictionaries_or_offsets():
    characters = [{"c": "x"}, {"c": "y"}]
    few = [{"c": "alpha"}, {"c": "beta"}, {"c": "alpha"}]
    many = [{"c": f"value-{index}"} for index in range(300)]
    for records in (characters, few, many):
        encoded = bjd.encode(records, tables=True)
        assert is_table(encoded)
        assert bjd.decode(encoded) == records
    assert b"C" in bjd.encode(characters, tables=True)[:8]
    assert b"[$S#" in bjd.encode(few, tables=True)


def test_an_offset_index_wide_enough_for_every_record():
    # Mostly empty text across more than 255 distinct values: far more records than text bytes.
    records = [{"c": str(index)} for index in range(300)] + [{"c": ""}] * 70_000
    assert bjd.decode(bjd.encode(records, tables=True)) == records


def test_integer_columns_narrow_to_their_range():
    assert bjd.encode([{"a": 1}, {"a": 2}], tables=True) == b"[${U\x01aU}#U\x02\x01\x02"
    assert bjd.encode([{"a": -1}, {"a": 300}], tables=True)[:7] == b"[${U\x01aI"
    assert bjd.encode([{"a": 1}, {"a": 2}], tables=True, prefer="speed")[:7] == b"[${U\x01aL"


def test_tables_nested_in_documents():
    document = {"rows": RECORDS, "count": 3}
    assert bjd.decode(bjd.encode(document, tables=True)) == document


def test_decoded_tables_with_numpy_are_structured_arrays():
    np = pytest.importorskip("numpy")
    decoded = bjd.decode(bjd.encode(RECORDS, tables=True), numpy=True)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype.names == ("id", "value", "ok", "name", "pos", "v", "none")
    assert decoded["id"].tolist() == [1, 2, 300]
    assert decoded["pos"]["y"].tolist() == [-1, -2, -3]
    assert decoded["v"].tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.5]]


STRUCTURED = [
    ("id", "<u4"),
    ("value", "<f8"),
    ("ok", "?"),
    ("name", "<U8"),
    ("pos", [("x", "<i2"), ("y", "<f4")]),
    ("v", "<f2", (3,)),
]


@pytest.mark.parametrize("order", ["row", "column"])
def test_structured_arrays_are_tables(order):
    np = pytest.importorskip("numpy")
    values = np.array(
        [(1, 0.5, True, "a", (1, 1.5), (1, 2, 3)), (2, 1.5, False, "bcd", (2, 2.5), (4, 5, 6))], dtype=STRUCTURED
    )
    encoded = bjd.encode(values, tables=True, table_order=order)
    assert is_table(encoded)
    decoded = bjd.decode(encoded, numpy=True)
    for name in ("id", "value", "ok", "v"):
        assert np.array_equal(decoded[name], values[name])
        assert decoded[name].dtype == values[name].dtype
    assert decoded["name"].tolist() == ["a", "bcd"]
    assert decoded["pos"].tolist() == values["pos"].tolist()


def test_structured_arrays_are_objects_without_tables():
    np = pytest.importorskip("numpy")
    values = np.array([(1, (2, 3.0))], dtype=[("a", "<i4"), ("b", [("c", "u1"), ("d", "<f4")])])
    assert bjd.decode(bjd.encode(values)) == [{"a": 1, "b": {"c": 2, "d": 3.0}}]
    assert bjd.decode(bjd.encode(values[0])) == {"a": 1, "b": {"c": 2, "d": 3.0}}


def test_nd_structured_arrays_keep_their_shape():
    np = pytest.importorskip("numpy")
    values = np.zeros((2, 3), dtype=[("a", "<i4"), ("b", "<f8")])
    values["a"] = np.arange(6).reshape(2, 3)
    encoded = bjd.encode(values, tables=True)
    assert is_table(encoded)
    decoded = bjd.decode(encoded, numpy=True)
    assert decoded.shape == (2, 3)
    assert np.array_equal(decoded["a"], values["a"])
    assert [[record["a"] for record in row] for row in bjd.decode(encoded)] == values["a"].tolist()
