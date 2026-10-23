# bjdata-serpent

[BJData](https://github.com/NeuroJSON/bjdata) (Binary JData) for Python, on the [serpent](https://github.com/nebkat/cpp-serpent) C++ library: draft 3 in full, and draft 4 Structure-of-Arrays tables.

```sh
pip install bjdata-serpent            # numpy is optional: pip install 'bjdata-serpent[numpy]'
```

```python
import bjdata_serpent as bjd

data = bjd.encode({"port": 4411, "samples": [1, 2, 3]})
bjd.decode(data)                       # {'port': 4411, 'samples': [1, 2, 3]}
bjd.notation(data)                     # '[{][U][4][port][u][4411][U][7][samples][[][U][1]...'
bjd.to_json(data)                      # '{"port":4411,"samples":[1,2,3]}'
```

## API

| | |
|---|---|
| `encode(obj, *, prefer="size", sort_keys=False, default=None, tables=False, table_order="row")` | one document, as bytes |
| `decode(data, *, numpy=False, object_hook=None, object_pairs_hook=None)` | the value of exactly one document |
| `decode_prefix(data, ...)` | the first value in `data`, and how many bytes it took |
| `dump(obj, fp, ...)` / `load(fp, ...)` | the same, to and from a binary file |
| `validate(data)` | raises `DecodeError` unless `data` is one well-formed value |
| `notation(data)` / `to_json(data, *, indent=0)` | what the bytes say, for reading |

`DecodeError` and `EncodeError` are `ValueError`s. A `DecodeError` carries the byte `offset` and serpent's name for what went wrong as `code` (`"unexpected_end"`, `"trailing_data"`, ...).

### Size or speed

`prefer="size"` (the default) writes each number under the narrowest marker that holds it exactly - `200` is `U`, `1.5` a two-byte half - and leaves containers uncounted. `prefer="speed"` writes numbers at their own width and counts every container, so a reader sizes it once. Either decodes the same.

### What becomes what

| Python | BJData | back |
|---|---|---|
| `None`, `bool` | `Z`, `T`/`F` | the same |
| `int` | the narrowest integer marker, or `H` digits beyond 64 bits | `int` |
| `float` | `h`/`d`/`D`, narrowed only when exact | `float` |
| `Decimal` | `H` | `Decimal`, or `int` for an integer |
| `str` | `S` | `str` |
| `bytes`, `bytearray`, byte `memoryview` | `[$B#` | `bytes` |
| `list`, `tuple`, other sequences | `[` | `list` |
| `dict`, other mappings | `{` | `dict` |
| numpy array, `array.array`, numeric `memoryview` | `[$T#n` typed, `[$T#[dims]` N-D | `list`, or with `numpy=True` an array of the same dtype and shape |
| numpy scalar | its own marker for speed, the narrowest for size | `int`/`float` |
| list of like dicts, structured numpy array (`tables=True`) | `[${schema}#n` table (`{${` with `table_order="column"`) | list of dicts, or with `numpy=True` a structured array |

What a document decodes to never depends on what is installed: packed arrays are lists unless `numpy=True` is asked for.

### Tables

With `tables=True`, a list of two or more dicts that share their keys - every value of a key being one kind a schema can say: booleans, integers, reals, text, `None`, nested dicts of the same, or lists of one length - is written as a Structure-of-Arrays table: the keys once, then the records back to back with no markers. Integer columns narrow to their range; text columns become single characters, a dictionary of up to 255 values, or a table of offsets. Anything else is written as an array of objects, as it would be without the option.

```python
rows = [{"id": i, "ok": i % 2 == 0, "station": f"s-{i % 4}"} for i in range(1000)]
len(bjd.encode(rows))                  # 28746
len(bjd.encode(rows, tables=True))     # 4053
```

## Compared with `bjdata`

[`bjdata`](https://pypi.org/project/bjdata/) (NeuroJSON's pybj) reads and writes the same format, and the two read each other's output. This one:

- has no required dependencies: numpy is used only when asked for, through its Python API, so there is no numpy ABI to build or run against;
- is one implementation, the extension, with no pure-Python fallback to drift from it;
- writes 2-2.6x faster and a few percent smaller, and reads about as fast;
- writes tables from plain lists of dicts as well as from structured arrays;
- ships wheels.

It does not write big-endian documents (pybj's `islittle=False`), it nests at most 32 deep, and it rejects typed containers of strings (`[$S#`), as serpent's grammar is strict.

## Building

The two C++ libraries are submodules:

```sh
git clone --recursive https://github.com/nebkat/python-bjdata-serpent
pip install '.[test,numpy]' && pytest
```

A C++23 compiler is needed: GCC 14, Clang 17, Apple Clang 15, or MSVC 19.40.
