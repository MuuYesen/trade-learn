"""Loading an oracle must restore the caller's imports on success and failure."""

import sys
import types

import pytest

from scripts import build_golden
from tradelearn.core.errors import GoldenDataError


@pytest.mark.parametrize("fails", [False, True])
def test_reference_query_restores_import_state(monkeypatch, tmp_path, fails):
    reference = tmp_path / "reference" / "tradelearn_1x"
    query = reference / "query"
    query.mkdir(parents=True)
    (query / "query.py").write_text("# required oracle layout\n")
    (query / "__init__.py").write_text(
        "raise RuntimeError('broken oracle')\n" if fails else
        "class Query:\n    source = 'isolated oracle'\n"
    )
    monkeypatch.setattr(build_golden, "REFERENCE", reference)
    monkeypatch.setattr(build_golden, "_module_available", lambda _: False)
    package_name = build_golden.REFERENCE_TDX_PACKAGE
    existing_bridge = types.ModuleType(package_name)
    original_quotes = object()
    existing_bridge.quotes = original_quotes
    monkeypatch.setitem(sys.modules, package_name, existing_bridge)
    names = ("tradelearn", "yfinance", "tvDatafeed", package_name,
             build_golden.REFERENCE_TDX_MODULE)
    before = {name: sys.modules.get(name) for name in names}
    before_path = list(sys.path)
    if fails:
        with pytest.raises(GoldenDataError, match="broken oracle"):
            build_golden.load_reference_query(allow_provider_stubs=True)
    else:
        loaded = build_golden.load_reference_query(allow_provider_stubs=True)
        assert loaded.source == "isolated oracle"
    assert sys.path == before_path
    assert {name: sys.modules.get(name) for name in names} == before
    assert existing_bridge.quotes is original_quotes
