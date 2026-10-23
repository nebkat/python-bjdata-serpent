"""BJData for Python, on serpent.

    >>> import bjdata_serpent as bjd
    >>> bjd.encode({"port": 4411})
    b'{U\\x04portu;\\x11}'
    >>> bjd.decode(_)
    {'port': 4411}

Packed and N-D arrays decode to (nested) lists unless ``numpy=True`` is given, so what a
document decodes to never depends on what happens to be installed.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from ._errors import DecodeError, EncodeError
from ._core import decode, decode_prefix, encode, notation, to_json, validate

try:
    __version__ = _version("bjdata-serpent")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "DecodeError",
    "EncodeError",
    "decode",
    "decode_prefix",
    "dump",
    "encode",
    "load",
    "notation",
    "to_json",
    "validate",
]


def dump(obj, fp, **options):
    """Writes ``encode(obj, **options)`` to a binary file."""
    fp.write(encode(obj, **options))


def load(fp, **options):
    """Reads a binary file to its end and decodes it as one document."""
    return decode(fp.read(), **options)
