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
