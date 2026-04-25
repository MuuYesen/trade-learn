"""Tests for pygwalker report exploration."""

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
    monkeypatch.delitem(__import__("sys").modules, "pygwalker", raising=False)

    with pytest.raises(ImportError, match=r"trade-learn\[lab\]"):
        Reporter({"trades": pd.DataFrame()}).explore()
