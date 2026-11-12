import subprocess
import sys

import pytest


def test_data_facade_and_custom_tushare_transport_do_not_import_requests():
    code = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'requests': raise AssertionError('unexpected requests import')
sys.meta_path.insert(0, Block())
from tradelearn.data import TushareProvider
TushareProvider(transport=lambda *args: None)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_requests_has_actionable_error_without_secret(monkeypatch):
    from tradelearn.data import TushareProvider

    monkeypatch.setitem(sys.modules, "requests", None)
    with pytest.raises(ImportError, match=r"trade-learn\[research\]") as error:
        TushareProvider(token="SECRET").history_ohlc(
            "600000.SH", start="2024-01-01", end="2024-01-04"
        )
    assert "SECRET" not in str(error.value)


def test_unrelated_custom_transport_import_error_stays_redacted():
    from tradelearn.data import TushareProvider

    def transport(*args):
        raise ImportError("custom SECRET")

    with pytest.raises(ConnectionError) as error:
        TushareProvider(transport=transport).history_ohlc(
            "600000.SH", start="2024-01-01", end="2024-01-04"
        )
    assert "SECRET" not in str(error.value)
