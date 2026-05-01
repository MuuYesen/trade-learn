from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import tradelearn as tl
import tradelearn.engine as bt
import tradelearn.lite as lite
from tradelearn.lite import Backtest
from tradelearn.lite import Strategy as LiteStrategy


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
    )


def test_engine_uses_vendor_indicator_namespaces_directly() -> None:
    seen: dict[str, float] = {}

    class VendorStrategy(bt.Strategy):
        def __init__(self) -> None:
            self.sma = bt.pta.SMA(self.data.close, length=2)
            self.tdx_ma = bt.tdx.MA(self.data.close, N=2)
            self.tv_sma = bt.tv.SMA(self.data.close, length=2)
            self.macd = bt.pta.MACD(self.data.close, fast=2, slow=3, signal=2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["sma"] = self.sma[0]
                seen["tdx_ma"] = self.tdx_ma[0]
                seen["tv_sma"] = self.tv_sma[0]
                seen["macd"] = self.macd.macd[0]

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.DataFeed(_bars()))
    cerebro.addstrategy(VendorStrategy)
    cerebro.run()

    assert seen["sma"] == 11.5
    assert seen["tdx_ma"] == 11.5
    assert seen["tv_sma"] == 11.5
    assert "macd" in seen


def test_lite_uses_vendor_indicator_namespaces_directly() -> None:
    seen: dict[str, float] = {}

    class VendorStrategy(LiteStrategy):
        def init(self) -> None:
            self.sma = tl.pta.SMA(self.data.close, length=2)
            self.tdx_ma = tl.tdx.MA(self.data.close, N=2)
            self.tv_sma = tl.tv.SMA(self.data.close, length=2)
            self.macd = tl.pta.MACD(self.data.close, fast=2, slow=3, signal=2)
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["sma"] = self.sma[0]
                seen["tdx_ma"] = self.tdx_ma[0]
                seen["tv_sma"] = self.tv_sma[0]
                seen["macd"] = self.macd.macd[0]

    Backtest(_bars(), VendorStrategy, cash=1000.0).run()

    assert seen["sma"] == 11.5
    assert seen["tdx_ma"] == 11.5
    assert seen["tv_sma"] == 11.5
    assert "macd" in seen


def test_engine_no_longer_exports_short_ind_alias() -> None:
    assert not hasattr(bt, "ind")
    assert "indicators" not in bt.__all__


def test_root_and_engine_share_vendor_indicator_namespaces() -> None:
    assert tl.pta is bt.pta
    assert tl.tdx is bt.tdx
    assert tl.tv is bt.tv
    assert hasattr(tl.pta, "__path__")
    assert "pta" in tl.__all__
    assert "talib" in tl.__all__
    assert "tdx" in tl.__all__
    assert "tv" in tl.__all__
    assert "pta" in tl.ta.__all__
    assert "tv" in tl.ta.__all__
    assert "pta" in bt.__all__
    assert "talib" in bt.__all__
    assert "tdx" in bt.__all__
    assert "tv" in bt.__all__


def test_vendor_indicator_namespaces_keep_init_files_as_facades() -> None:
    assert tl.pta.SMA._func.__module__ == "tradelearn.indicators.pta.pandas_ta_adapter"
    assert tl.tdx.MA._func.__module__ == "tradelearn.indicators.tdx.mytt_adapter"
    assert tl.tv.SMA._func.__module__ == "tradelearn.indicators.tv.pynecore_adapter"


def test_vendor_indicator_namespaces_are_case_compatible() -> None:
    close = _bars()["close"]

    assert tl.pta.SMA is tl.pta.sma
    assert tl.pta.EMA is tl.pta.ema
    assert tl.pta.RSI is tl.pta.rsi
    assert tl.pta.MACD is tl.pta.macd
    assert tl.pta.ATR is tl.pta.atr

    pd.testing.assert_series_equal(tl.tdx.MA(close, N=2), tl.tdx.ma(close, n=2))
    pd.testing.assert_series_equal(tl.tdx.RSI(close, N=2), tl.tdx.rsi(close, n=2))
    pd.testing.assert_frame_equal(tl.tdx.MACD(close), tl.tdx.macd(close))

    assert tl.tv.SMA is tl.tv.sma
    assert tl.tv.RSI is tl.tv.rsi
    assert tl.tv.MACD is tl.tv.macd


def test_real_talib_namespace_is_not_pandas_ta_fallback() -> None:
    try:
        import talib as _talib  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="requires TA-Lib"):
            import importlib

            importlib.import_module("tradelearn.indicators.talib")
        return

    assert tl.talib.SMA._func.__module__ == "tradelearn.indicators.talib.ta_lib_adapter"


def test_vendor_indicator_adapter_files_are_named_by_backend() -> None:
    root = Path(__file__).resolve().parents[3] / "tradelearn" / "indicators"

    assert (root / "pta" / "pandas_ta_adapter.py").exists()
    assert (root / "talib" / "ta_lib_adapter.py").exists()
    assert not (root / "talib" / "pandas_ta_adapter.py").exists()
    assert (root / "tdx" / "mytt_adapter.py").exists()
    assert (root / "tv" / "pynecore_adapter.py").exists()
    assert not (root / "tdx" / "mytt.py").exists()
    assert not (root / "tdx" / "tdx30.py").exists()


def test_lite_and_engine_share_vendor_indicator_namespaces() -> None:
    assert lite.ta is tl.ta
    assert lite.pta is bt.pta
    assert lite.tdx is bt.tdx
    assert lite.tv is bt.tv
    assert "ta" in lite.__all__
    assert "pta" in lite.__all__
    assert "talib" in lite.__all__
    assert "tdx" in lite.__all__
    assert "tv" in lite.__all__


def test_lite_no_longer_exports_chain_ta_accessors() -> None:
    seen: dict[str, bool] = {}

    class VendorStrategy(LiteStrategy):
        def init(self) -> None:
            seen["data_has_ta"] = hasattr(self.data, "ta")
            seen["close_has_ta"] = hasattr(self.data.close, "ta")
            self.start_on_bar(2)

    Backtest(_bars(), VendorStrategy, cash=1000.0).run()

    assert seen == {"data_has_ta": False, "close_has_ta": False}
