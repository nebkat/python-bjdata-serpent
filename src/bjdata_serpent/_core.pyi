from collections.abc import Callable
from typing import Any, Literal

from typing_extensions import Buffer

def encode(
    obj: Any,
    *,
    prefer: Literal["size", "speed"] = "size",
    sort_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    tables: bool = False,
    table_order: Literal["row", "column"] = "row",
) -> bytes:
    """``obj`` as one BJData document.

    ``prefer="size"`` writes each number under the narrowest marker that holds it exactly and
    leaves containers uncounted; ``"speed"`` writes numbers at their own width and counts every
    container, so a reader sizes it once. Lists, tuples and other sequences are arrays; dicts and
    other mappings with str keys are objects; bytes, bytearray and byte memoryviews are ``[$B``
    binary; ints too wide for 64 bits, and Decimals, are high-precision ``H``.

    Anything exporting a buffer of numbers - a numpy array or scalar, an ``array.array``, a
    memoryview - is a typed array, counted by its dimensions when it has more than one, in one
    copy when its bytes are already in order.

    ``tables=True`` writes a list of two or more dicts with the same keys, whose values a schema
    can say, as a draft 4 Structure-of-Arrays table, and a structured numpy array likewise;
    ``table_order="column"`` lays one out field by field. ``default`` is called with anything
    else and returns something that can be encoded, as for ``json.dumps``.

    Raises TypeError for a value that cannot be encoded, EncodeError for one nested more than 32
    deep (or containing itself).
    """

def decode(
    data: Buffer,
    *,
    numpy: bool = False,
    object_hook: Callable[[dict[str, Any]], Any] | None = None,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
) -> Any:
    """The value a whole BJData document holds.

    Packed and N-D arrays are (nested) lists, and tables lists of dicts, unless ``numpy=True``,
    which makes them numpy arrays of their own dtype and shape and tables structured arrays.
    ``[$B`` binary is bytes; ``H`` is an int when it is an integer and a Decimal otherwise.

    Raises DecodeError, with the byte offset, for anything but exactly one well-formed value.
    """

def decode_prefix(
    data: Buffer,
    *,
    numpy: bool = False,
    object_hook: Callable[[dict[str, Any]], Any] | None = None,
    object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
) -> tuple[Any, int]:
    """The value the document starts with, and how many bytes it took."""

def validate(data: Buffer) -> None:
    """Raises DecodeError unless ``data`` is exactly one well-formed BJData value."""

def notation(data: Buffer) -> str:
    """The document's block notation - ``[{][U][4][port]...`` - for reading what bytes say."""

def to_json(data: Buffer, *, indent: int = 0) -> str:
    """The document rendered as JSON, N-D arrays nested."""
