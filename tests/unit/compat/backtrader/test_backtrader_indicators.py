from __future__ import annotations

import math

import pandas as pd

import tradelearn.compat.backtrader as bt


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [8.0, 9.0, 10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            utc=True,
        ),
    )


def test_backtrader_indicators_exports_common_twenty_aliases() -> None:
    expected = {
        "SMA",
        "SimpleMovingAverage",
        "MovingAverageSimple",
        "EMA",
        "ExponentialMovingAverage",
        "WMA",
        "WeightedMovingAverage",
        "RSI",
        "RelativeStrengthIndex",
        "MACD",
        "BollingerBands",
        "BBands",
        "Highest",
        "Lowest",
        "ATR",
        "AverageTrueRange",
        "TrueRange",
        "CrossOver",
        "CrossUp",
        "CrossDown",
        "Stochastic",
    }

    assert expected.issubset(set(bt.indicators.SUPPORTED_INDICATOR_ALIASES))
    assert len(bt.indicators.SUPPORTED_INDICATOR_ALIASES) >= 20
    assert all(callable(getattr(bt.indicators, name)) for name in expected)


def test_sma_line_tracks_backtrader_current_bar_index() -> None:
    class RecordingStrategy(bt.Strategy):
        def __init__(self) -> None:
            self.sma = bt.indicators.SMA(self.data.close, period=3)
            self.values: list[float] = []

        def next(self) -> None:
            self.values.append(self.sma[0])

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=bars(), name="daily"))
    cerebro.addstrategy(RecordingStrategy)

    [strategy] = cerebro.run()

    assert math.isnan(strategy.values[0])
    assert math.isnan(strategy.values[1])
    assert strategy.values[2:] == [11.0, 12.0, 13.0]


def test_multiline_indicators_expose_backtrader_style_lines() -> None:
    data = bt.feeds.PandasData(dataname=bars())
    macd = bt.indicators.MACD(data.close, fast=2, slow=3, signal=2)
    bands = bt.indicators.BollingerBands(data.close, period=3, devfactor=2.0)

    data._advance(4)

    assert math.isfinite(macd.macd[0])
    assert math.isfinite(macd.signal[0])
    assert math.isfinite(macd.histo[0])
    assert bands.top[0] > bands.mid[0] > bands.bot[0]
