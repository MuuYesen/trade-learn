"""Actual bounded example lines retain formulas and cannot observe future bars."""

import importlib

import numpy as np
import pandas as pd
import pytest

from examples.lite._indicators import EMA, MACD, Rolling, TrueRange
from tradelearn.lite import Backtest, Strategy


def bars():
    close = np.array([10 + i * 0.1 + np.sin(i) for i in range(45)])
    return pd.DataFrame(
        dict(
            open=close - 0.1,
            high=close + 0.5,
            low=close - 0.6,
            close=close,
            volume=np.arange(45) + 100.0,
        ),
        index=pd.date_range("2026-01-01", periods=45, tz="UTC"),
    )


class Probe(Strategy):
    def init(self):
        self.ema = EMA(self.data.close, period=5)
        self.rolling = Rolling(self.data.close, period=5)
        self.macd = MACD(self.data.close, fast=3, slow=6, signal=2)
        self.atr = Rolling(TrueRange(self.data), period=5).mean
        self.trace = []

    def next(self):
        self.trace.append(
            [
                self.ema[0],
                self.rolling.mean[0],
                self.rolling.upper[0],
                self.macd.dif[0],
                self.macd.dea[0],
                self.atr[0],
            ]
        )


def test_streaming_example_lines_match_previous_pandas_formulas_and_prefix():
    data = bars()
    first = Backtest(data, Probe).run().strategy
    close = data.close
    diff = close.ewm(span=3, adjust=False).mean() - close.ewm(span=6, adjust=False).mean()
    tr = pd.concat(
        [data.high - data.low, (data.high - close.shift()).abs(), (data.low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    expected = np.column_stack(
        [
            close.ewm(span=5, adjust=False).mean(),
            close.rolling(5).mean(),
            close.rolling(5).mean() + 2 * close.rolling(5).std(ddof=0),
            diff,
            diff.ewm(span=2, adjust=False).mean(),
            tr.rolling(5).mean(),
        ]
    )
    np.testing.assert_allclose(first.trace, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    changed = data.copy()
    changed.iloc[20:, :4] *= 3
    second = Backtest(changed, Probe).run().strategy
    np.testing.assert_array_equal(np.array(first.trace)[:20], np.array(second.trace)[:20])
    assert len(first.rolling.values) == first.rolling.values.maxlen == 5


@pytest.mark.parametrize(
    "module,name",
    [
        ("01_ema_cross", "EMA_Cross_Strategy"),
        ("02_macd", "MACD_Volume_Strategy"),
        ("03_sma_cross", "LiteSmaCross"),
        ("04_bollinger", "LiteBollBandCross"),
        ("05_macd", "LiteMACDCross"),
        ("06_grid_trade", "LiteGridTrade"),
        ("07_turtle", "LiteAdvancedTurtle"),
    ],
)
def test_all_lite_examples_use_delivered_lines(module, name):
    strategy = getattr(importlib.import_module("examples.lite." + module), name)
    result = Backtest(bars(), strategy, cash=10_000).run()
    assert np.isfinite(result["final_value"])
