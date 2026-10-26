"""Malformed documents, the document-level helpers, and serpent's cross-implementation corpus."""

import io
import json
import pathlib

import pytest

import bjdata_serpent as bjd

FIXTURES = pathlib.Path(__file__).parent.parent / "extern" / "cpp-serpent" / "test" / "fixtures"
CORPUS = sorted(FIXTURES.glob("*.bjd")) + sorted(FIXTURES.glob("soa/*.bjd"))


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.stem)
def test_dart_bjdata_encodings_decode_to_their_json(path):
    expected = json.loads(path.with_suffix(".json").read_text())
    assert bjd.decode(path.read_bytes()) == expected
    for prefer in ("size", "speed"):
        assert bjd.decode(bjd.encode(expected, prefer=prefer, tables=True)) == expected


@pytest.mark.parametrize("path", CORPUS, ids=lambda path: path.stem)
def test_every_truncation_is_a_decode_error(path):
    document = path.read_bytes()
    for end in range(len(document)):
        with pytest.raises(bjd.DecodeError):
            bjd.decode(document[:end])


@pytest.mark.parametrize(
    "document, code, offset",
    [
        (b"", "unexpected_end", 0),
        (b"ZZ", "trailing_data", 1),
        (b"X", "unexpected_marker", 0),
        (b"[U", "unexpected_end", 2),
        (b"SU\x05ab", "unexpected_end", 3),
        (b"E", "extension_unsupported", 0),
        (b"[" * 40 + b"]" * 40, "depth_exceeded", None),
    ],
)
def test_decode_errors_say_what_and_where(document, code, offset):
    with pytest.raises(bjd.DecodeError) as raised:
        bjd.decode(document)
    assert raised.value.code == code
    if offset is not None:
        assert raised.value.offset == offset
    assert isinstance(raised.value, ValueError)


def test_text_that_is_not_utf8_is_a_decode_error():
    with pytest.raises(bjd.DecodeError, match="not UTF-8") as raised:
        bjd.decode(b"SU\x02\xff\xfe")
    assert raised.value.code is None
    with pytest.raises(bjd.DecodeError):
        bjd.decode(b"{U\x01\xffZ}")


def test_decoding_str_is_a_type_error():
    with pytest.raises(TypeError, match="bytes, not str"):
        bjd.decode("Z")


def test_any_bytes_like_object_decodes():
    for data in (b"U\x05", bytearray(b"U\x05"), memoryview(b"xU\x05")[1:]):
        assert bjd.decode(data) == 5


def test_decode_prefix_reads_one_value_of_several():
    stream = bjd.encode({"a": 1}) + bjd.encode([2]) + bjd.encode("three")
    values = []
    while stream:
        value, end = bjd.decode_prefix(stream)
        values.append(value)
        stream = stream[end:]
    assert values == [{"a": 1}, [2], "three"]
    with pytest.raises(bjd.DecodeError):
        bjd.decode_prefix(b"[U")


def test_validate():
    bjd.validate(b"[U\x01]")
    with pytest.raises(bjd.DecodeError):
        bjd.validate(b"[U\x01")


def test_notation_and_json():
    document = bjd.encode({"name": "x", "n": [1, 2]})
    assert bjd.notation(document) == "[{][U][4][name][S][U][1][x][U][1][n][[][U][1][U][2][]][}]"
    assert bjd.to_json(document) == '{"name":"x","n":[1,2]}'
    assert json.loads(bjd.to_json(document, indent=2)) == {"name": "x", "n": [1, 2]}


def test_dump_and_load():
    stream = io.BytesIO()
    bjd.dump({"a": [1, 2]}, stream, prefer="speed")
    stream.seek(0)
    assert bjd.load(stream) == {"a": [1, 2]}


# A table's records are objects one level inside it, so a table reaches the limit one array sooner.
@pytest.mark.parametrize("inner, deepest", [(b"[$i#U\x02\x01\x02", 31), (b"[${U\x01aU}#U\x02\x01\x02", 30), (b"[U\x01]", 31)],
                         ids=["typed", "table", "plain"])
def test_validate_and_decode_agree_at_the_depth_limit(inner, deepest):
    for depth, accepted in ((deepest, True), (deepest + 1, False)):
        document = b"[" * depth + inner + b"]" * depth
        if accepted:
            bjd.validate(document)
            bjd.decode(document)
            continue
        for check in (bjd.validate, bjd.decode):
            with pytest.raises(bjd.DecodeError) as raised:
                check(document)
            assert raised.value.code == "depth_exceeded"


def test_hostile_nesting_is_refused_without_recursing_into_it():
    for document in (b"[" * 1_000_000, b"{U\x01a" * 100_000, b"[#U\x01" * 1_000_000):
        with pytest.raises(bjd.DecodeError) as raised:
            bjd.decode(document)
        assert raised.value.code == "depth_exceeded"


@pytest.mark.parametrize("numpy", [False, True])
def test_counts_the_bytes_cannot_hold_are_refused_before_anything_is_allocated(numpy):
    for document in (b"[$l#L" + (2**40).to_bytes(8, "little"), b"[#L" + (2**40).to_bytes(8, "little") + b"U\x01",
                     b"[$D#[L" + (2**30).to_bytes(8, "little") + b"L" + (2**30).to_bytes(8, "little") + b"]",
                     b"[${U\x01aZ}#L" + (2**40).to_bytes(8, "little")):
        with pytest.raises(bjd.DecodeError):
            bjd.decode(document, numpy=numpy)


def test_a_hook_that_raises_is_its_own_error():
    def refuse(_):
        raise LookupError("no")

    with pytest.raises(LookupError):
        bjd.decode(bjd.encode({"a": 1}), object_hook=refuse)
    with pytest.raises(bjd.DecodeError, match="unexpected end"):
        bjd.decode(bjd.encode({"a": 1})[:-1], object_hook=refuse)


def test_decode_prefix_validates_only_the_leading_value():
    assert bjd.decode_prefix(b"U\x01\xff\xff") == (1, 2)
    with pytest.raises(bjd.DecodeError) as raised:
        bjd.decode_prefix(b"[U\x01")
    assert raised.value.code == "unexpected_end"
