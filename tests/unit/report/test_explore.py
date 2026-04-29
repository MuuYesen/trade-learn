"""Tests for pygwalker report exploration."""

import builtins
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from tradelearn.report import Reporter


def test_reporter_explore_calls_pygwalker_with_trades(monkeypatch) -> None:
    """Reporter.explore delegates to pygwalker on the trades table."""
    trades = pd.DataFrame({"pnl": [1.0, -0.5]})
    fake_pygwalker = ModuleType("pygwalker")
    calls = []

    def walk(frame):
        calls.append(frame)
        return "walker"

    fake_pygwalker.walk = walk
    monkeypatch.setitem(__import__("sys").modules, "pygwalker", fake_pygwalker)

    result = Reporter(SimpleNamespace(trades=trades)).explore()

    assert result == "walker"
    assert calls == [trades]


def test_reporter_explore_has_install_hint_when_missing(monkeypatch) -> None:
    """Reporter.explore reports how to install pygwalker when it is missing."""
    real_import = builtins.__import__

    def missing_pygwalker(name, *args, **kwargs):
        if name == "pygwalker":
            raise ImportError("missing pygwalker")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pygwalker)

    with pytest.raises(ImportError, match=r"trade-learn\[lab\]"):
        Reporter({"trades": pd.DataFrame({"pnl": [1.0]})}).explore()


def test_reporter_explore_rejects_empty_trades(monkeypatch) -> None:
    """Reporter.explore avoids pygwalker empty-table failures."""
    fake_pygwalker = ModuleType("pygwalker")
    fake_pygwalker.walk = lambda frame: "walker"
    monkeypatch.setitem(__import__("sys").modules, "pygwalker", fake_pygwalker)

    with pytest.raises(ValueError, match="at least one trade row"):
        Reporter({"trades": pd.DataFrame()}).explore()
