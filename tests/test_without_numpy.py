"""Nothing needs numpy but asking for it, whether or not it is installed."""

import subprocess
import sys

SCRIPT = r"""
import sys
sys.modules["numpy"] = None  # any import of it now fails

import bjdata_serpent as bjd

document = bjd.encode({"a": [1, 2.5, "x", b"\x00"], "rows": [{"k": 1}, {"k": 2}]}, tables=True)
assert bjd.decode(document) == {"a": [1, 2.5, "x", b"\x00"], "rows": [{"k": 1}, {"k": 2}]}
assert bjd.decode(b"[$l#U\x02\x01\x00\x00\x00\x02\x00\x00\x00") == [1, 2]
assert bjd.decode(b"[$U#[U\x02U\x02]\x01\x02\x03\x04") == [[1, 2], [3, 4]]
try:
    bjd.decode(b"[$l#U\x01\x01\x00\x00\x00", numpy=True)
except ImportError:
    pass
else:
    raise AssertionError("numpy=True without numpy")
assert sys.modules["numpy"] is None
"""


def test_without_numpy():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
