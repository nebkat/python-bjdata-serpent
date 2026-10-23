"""Scalars and containers: the bytes they become, and what those decode back to."""

import math
from collections import OrderedDict
from decimal import Decimal

import pytest

import bjdata_serpent as bjd


@pytest.mark.parametrize(
    "value, encoded",
    [
        (None, b"Z"),
        (True, b"T"),
        (False, b"F"),
        (0, b"U\x00"),
        (255, b"U\xff"),
        (-1, b"i\xff"),
        (-128, b"i\x80"),
        (256, b"u\x00\x01"),
        (-129, b"I\x7f\xff"),
        (65536, b"m\x00\x00\x01\x00"),
        (2**63, b"M" + (2**63).to_bytes(8, "little")),
        (-(2**63), b"L" + (2**63).to_bytes(8, "little")),
        (1.0, b"h\x00\x3c"),
        (0.1, b"D" + bytes.fromhex("9a9999999999b93f")),
        ("", b"SU\x00"),
        ("a", b"SU\x01a"),
        ("héllo", b"SU\x06h\xc3\xa9llo"),
        (b"\x01\x02", b"[$B#U\x02\x01\x02"),
        ([], b"[]"),
        ({}, b"{}"),
        ([1, "a"], b"[U\x01SU\x01a]"),
        ({"a": None}, b"{U\x01aZ}"),
    ],
)
def test_encodes_as_small_as_it_can(value, encoded):
    assert bjd.encode(value) == encoded
    assert bjd.decode(encoded) == value


@pytest.mark.parametrize(
    "value, encoded",
    [
        (1, b"L" + (1).to_bytes(8, "little")),
        (1.0, b"D" + bytes.fromhex("000000000000f03f")),
        ([], b"[#U\x00"),
        ([True, None], b"[#U\x02TZ"),
        ({"a": 1}, b"{#U\x01U\x01aL" + (1).to_bytes(8, "little")),
    ],
)
def test_encodes_for_speed(value, encoded):
    assert bjd.encode(value, prefer="speed") == encoded
    assert bjd.decode(encoded) == value


@pytest.mark.parametrize(
    "value",
    [
        0,
        2**64 - 1,
        -(2**63),
        2**64,
        -(2**63) - 1,
        10**40,
        -(10**40),
        0.5,
        -0.0,
        1e300,
        5e-324,
        float("inf"),
        float("-inf"),
        3.141592653589793,
        "x" * 100_000,
        "☃\U0001f40d",
        b"",
        bytes(range(256)) * 10,
        [[[[]]]],
        {"nested": {"deeper": [1, {"x": [None]}]}},
        list(range(-300, 300)),
    ],
)
@pytest.mark.parametrize("prefer", ["size", "speed"])
def test_round_trips(value, prefer):
    assert bjd.decode(bjd.encode(value, prefer=prefer)) == value


def test_nan_round_trips():
    assert math.isnan(bjd.decode(bjd.encode(float("nan"))))


def test_a_negative_zero_keeps_its_sign():
    assert math.copysign(1, bjd.decode(bjd.encode(-0.0))) == -1


def test_integers_wider_than_64_bits_are_high_precision_and_come_back_as_int():
    encoded = bjd.encode(2**64)
    assert encoded == b"HU\x14" + b"18446744073709551616"
    assert type(bjd.decode(encoded)) is int


def test_decimals_are_high_precision():
    value = Decimal("3.14159265358979323846264338327950288")
    encoded = bjd.encode(value)
    assert encoded[:1] == b"H"
    assert bjd.decode(encoded) == value
    assert bjd.decode(bjd.encode(Decimal(10))) == 10
    assert math.isinf(bjd.decode(bjd.encode(Decimal("Infinity"))))


def test_floats_narrow_only_when_exact():
    assert bjd.encode(1.5)[:1] == b"h"
    assert bjd.encode(65504.0)[:1] == b"h"
    assert bjd.encode(1e10)[:1] == b"d"
    assert bjd.encode(0.1)[:1] == b"D"


def test_bytes_like_are_binary():
    for value in (bytearray(b"ab"), memoryview(b"ab")):
        assert bjd.encode(value) == b"[$B#U\x02ab"
    assert bjd.decode(b"[$B#U\x02ab") == b"ab"


def test_tuples_and_other_sequences_are_arrays():
    assert bjd.decode(bjd.encode((1, 2))) == [1, 2]
    assert bjd.decode(bjd.encode(range(3))) == [0, 1, 2]


def test_other_mappings_are_objects():
    assert bjd.decode(bjd.encode(OrderedDict(b=1, a=2))) == {"b": 1, "a": 2}


def test_sort_keys():
    assert bjd.encode({"b": 1, "a": 2}, sort_keys=True) == bjd.encode({"a": 2, "b": 1})


def test_keys_keep_their_order():
    assert list(bjd.decode(bjd.encode({"z": 1, "a": 2, "m": 3}))) == ["z", "a", "m"]


def test_keys_must_be_str():
    with pytest.raises(TypeError, match="keys must be str, not int"):
        bjd.encode({1: 2})


def test_unsupported_types_raise_type_error():
    with pytest.raises(TypeError, match="not BJData serializable"):
        bjd.encode(object())
    with pytest.raises(TypeError):
        bjd.encode({1, 2})


def test_default_replaces_what_cannot_be_encoded():
    assert bjd.decode(bjd.encode({1, 2}, default=sorted)) == [1, 2]
    assert bjd.decode(bjd.encode([complex(1, 2)], default=lambda c: [c.real, c.imag])) == [[1.0, 2.0]]


def test_default_that_never_settles_is_an_encode_error():
    with pytest.raises(bjd.EncodeError):
        bjd.encode(object(), default=lambda value: object())


def test_nesting_is_capped():
    deep = []
    for _ in range(33):
        deep = [deep]
    with pytest.raises(bjd.EncodeError, match="nesting deeper than 32"):
        bjd.encode(deep)
    shallow = []
    for _ in range(31):
        shallow = [shallow]
    assert bjd.decode(bjd.encode(shallow)) == shallow


def test_a_value_that_contains_itself_is_an_encode_error():
    loop = []
    loop.append(loop)
    with pytest.raises(bjd.EncodeError):
        bjd.encode(loop)


def test_strings_that_are_not_utf8_cannot_be_encoded():
    with pytest.raises(UnicodeEncodeError):
        bjd.encode("\ud800")


def test_prefer_and_table_order_are_checked():
    with pytest.raises(ValueError, match="prefer must be 'size' or 'speed'"):
        bjd.encode(1, prefer="fast")
    with pytest.raises(ValueError, match="table_order"):
        bjd.encode(1, table_order="diagonal")


def test_object_hook_and_object_pairs_hook():
    document = bjd.encode({"a": {"b": 1}})
    assert bjd.decode(document, object_pairs_hook=list) == [("a", [("b", 1)])]
    assert bjd.decode(document, object_hook=lambda d: sorted(d)) == ["a"]


def test_character_decodes_to_str():
    assert bjd.decode(b"Ca") == "a"


def test_noops_are_skipped():
    assert bjd.decode(b"[NU\x01NN]") == [1]
