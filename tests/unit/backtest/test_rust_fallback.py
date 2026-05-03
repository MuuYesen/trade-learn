from __future__ import annotations

import builtins

import pandas as pd

from tradelearn.engine import Cerebro, Strategy


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 110.0, 120.0],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
    )


def test_engine_run_falls_back_to_python_runtime_when_rust_extension_is_unavailable(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def import_without_tradelearn_rust(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tradelearn._rust":
            raise ImportError("tradelearn._rust unavailable")
        return original_import(name, globals, locals, fromlist, level)

    class Noop(Strategy):
        def next(self) -> None:
            pass

    monkeypatch.setattr(builtins, "__import__", import_without_tradelearn_rust)

    cerebro = Cerebro()
    cerebro.setcash(1234.0)
    cerebro.adddata(_bars())
    cerebro.addstrategy(Noop)

    [strategy] = cerebro.run()

    assert strategy.stats.summary["final_value"] == 1234.0
    assert strategy.stats.summary["bars"] == 3
