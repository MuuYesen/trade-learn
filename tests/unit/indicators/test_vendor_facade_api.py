from __future__ import annotations

import math

import pandas as pd

import tradelearn as tl
import tradelearn.engine as bt
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
            self.sma = bt.talib.SMA(self.data.close, timeperiod=2)
            self.tdx_ma = bt.tdx.MA(self.data.close, N=2)
            self.tv_rsi = bt.tv.RSI(self.data.close, length=2)
            self.macd = bt.talib.MACD(self.data.close, fastperiod=2, slowperiod=3, signalperiod=2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["sma"] = self.sma[0]
                seen["tdx_ma"] = self.tdx_ma[0]
                seen["tv_rsi"] = self.tv_rsi[0]
                seen["macd"] = self.macd.macd[0]

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.DataFeed(_bars()))
    cerebro.addstrategy(VendorStrategy)
    cerebro.run()

    assert seen["sma"] == 11.5
    assert seen["tdx_ma"] == 11.5
    assert math.isfinite(seen["tv_rsi"])
    assert "macd" in seen


def test_lite_uses_vendor_indicator_namespaces_directly() -> None:
    seen: dict[str, float] = {}

    class VendorStrategy(LiteStrategy):
        def init(self) -> None:
            self.sma = tl.talib.SMA(self.data.close, timeperiod=2)
            self.tdx_ma = tl.tdx.MA(self.data.close, N=2)
            self.tv_rsi = tl.tv.RSI(self.data.close, length=2)
            self.macd = tl.talib.MACD(self.data.close, fastperiod=2, slowperiod=3, signalperiod=2)
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["sma"] = self.sma[0]
                seen["tdx_ma"] = self.tdx_ma[0]
                seen["tv_rsi"] = self.tv_rsi[0]
                seen["macd"] = self.macd.macd[0]

    Backtest(_bars(), VendorStrategy, cash=1000.0).run()

    assert seen["sma"] == 11.5
    assert seen["tdx_ma"] == 11.5
    assert math.isfinite(seen["tv_rsi"])
    assert "macd" in seen


def test_engine_no_longer_exports_short_ind_alias() -> None:
    assert not hasattr(bt, "ind")
    assert "indicators" not in bt.__all__


def test_lite_no_longer_exports_chain_ta_accessors() -> None:
    seen: dict[str, bool] = {}

    class VendorStrategy(LiteStrategy):
        def init(self) -> None:
            seen["data_has_ta"] = hasattr(self.data, "ta")
            seen["close_has_ta"] = hasattr(self.data.close, "ta")
            self.start_on_bar(2)

    Backtest(_bars(), VendorStrategy, cash=1000.0).run()

    assert seen == {"data_has_ta": False, "close_has_ta": False}
