"""Only the two inspected official LF/CRLF source byte variants are accepted."""

import hashlib
import inspect
from pathlib import Path

import pytest

from tradelearn.indicators._kunquant_graph import (
    ALPHA158_SOURCE_SHA256,
    SOURCE_SHA256,
    _fixed_source_file,
)


@pytest.mark.parametrize(
    "module,canonical,windows",
    [
        (
            "Alpha101",
            SOURCE_SHA256,
            "6b974d42b4289ea6a8af8c7b26f752ee777a1caa697e190cd199fdda4537c5f3",
        ),
        (
            "Alpha158",
            ALPHA158_SOURCE_SHA256,
            "5afe30b56fbbe4fe5431a232153ed1c77dc25571b8786d4bae35be07a27a70b4",
        ),
    ],
)
def test_exact_official_newline_variants_and_source_drift_rejection(
    tmp_path, module, canonical, windows
):
    pytest.importorskip("KunQuant")
    from importlib import import_module

    upstream = import_module("KunQuant.predefined." + module)
    raw = Path(inspect.getfile(upstream)).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == canonical
    path = tmp_path / "source.py"
    path.write_bytes(raw)
    assert _fixed_source_file(path, canonical, windows) == canonical
    path.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert _fixed_source_file(path, canonical, windows) == windows
    path.write_bytes(raw + b"\n# changed source\n")
    with pytest.raises(ValueError, match="fixed qualification scope"):
        _fixed_source_file(path, canonical, windows)
