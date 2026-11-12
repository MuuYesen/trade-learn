"""PTA retirement must not leave hidden transitive runtime imports."""

import importlib
from pathlib import Path

import pytest


def test_legacy_vendor_tree_is_absent():
    import tradelearn

    root = Path(tradelearn.__file__).resolve().parent.parent
    assert not (root / "tradelearn/query/tec/pandas_ta").exists()


def test_direct_retired_backend_import_explains_migration():
    with pytest.raises(ImportError, match="PTA was retired"):
        importlib.import_module("tradelearn.indicators.pta")


def test_runtime_source_contains_no_pandas_ta_imports():
    import ast

    import tradelearn

    root = Path(tradelearn.__file__).resolve().parent.parent / "tradelearn/indicators"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(not x.name.startswith("pandas_ta") for x in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pandas_ta"), path
