from __future__ import annotations

import importlib
from typing import Type

import pandas as pd

import tradelearn.engine as bt
from tradelearn.lite import Backtest as LiteBacktest
from tradelearn.lite import Strategy as LiteStrategy


def _sample_data() -> pd.DataFrame:
    close = [
        10.0,
        10.2,
        10.5,
        10.1,
        9.8,
        10.4,
        11.2,
        12.1,
        11.4,
        10.6,
        9.9,
        10.7,
        11.6,
        12.8,
        13.3,
        12.4,
        11.2,
        10.3,
        9.7,
        10.5,
        11.8,
        13.0,
        14.1,
        13.2,
        12.0,
        11.0,
        10.2,
        11.4,
        12.6,
        13.8,
        12.7,
        11.5,
        10.4,
        9.6,
        10.8,
        12.0,
        13.4,
        14.5,
        13.6,
        12.3,
    ]
    return pd.DataFrame(
        {
            "open": [value - 0.1 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.6 for value in close],
            "close": close,
            "volume": [1000.0 + i * 20.0 for i in range(len(close))],
        },
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )


def _load_strategy(module_name: str, class_name: str) -> Type[LiteStrategy]:
    module = importlib.import_module(f"examples.lite.{module_name}")
    return getattr(module, class_name)


def _run_lite(strategy: Type[LiteStrategy], **params) -> tuple[float, int]:
    stats = LiteBacktest(
        _sample_data(),
        strategy,
        cash=10000.0,
        commission=0.0,
        trade_on_close=True,
    ).run(**params)
    return float(stats["Equity Final [$]"]), int(stats["# Trades"])


def _run_engine(strategy: type[bt.Strategy], **params) -> tuple[float, int]:
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.adddata(bt.feeds.PandasData(dataname=_sample_data(), name="demo"))
    cerebro.addstrategy(strategy, **params)
    cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.broker.setcash(10000.0)
    [result] = cerebro.run()
    assert result.stats is not None
    return float(result.stats.summary["final_value"]), len(result.stats.fills)


def test_tradelearn_1x_sma_example_runs_on_lite_and_engine_runtime() -> None:
    lite_strategy = _load_strategy("03_1x_sma_cross", "OneXSmaCross")

    class EngineSmaCross(bt.Strategy):
        params = (("fast", 10), ("slow", 20))

        def __init__(self) -> None:
            self.ma1 = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.ma2 = bt.indicators.SMA(self.data.close, period=self.p.slow)

        def next(self) -> None:
            if len(self.data.close) < 2:
                return
            crosses_up = self.ma1[-1] < self.ma2[-1] and self.ma1[0] > self.ma2[0]
            crosses_down = self.ma2[-1] < self.ma1[-1] and self.ma2[0] > self.ma1[0]
            if crosses_up:
                self.close()
                self.buy()
            elif crosses_down:
                self.close()
                self.sell()

    lite_value, lite_trades = _run_lite(lite_strategy)
    engine_value, engine_trades = _run_engine(EngineSmaCross)

    assert lite_value > 0
    assert lite_trades >= 0
    assert engine_value > 0
    assert engine_trades >= 0


def test_tradelearn_1x_macd_example_runs_on_lite_and_engine_runtime() -> None:
    lite_strategy = _load_strategy("05_1x_macd", "OneXMACDCross")

    class EngineMACDCross(bt.Strategy):
        params = (("s", 12), ("l", 26), ("m", 9), ("title", "Long"))

        def __init__(self) -> None:
            self.macd = bt.indicators.MACD(
                self.data,
                period_me1=self.p.s,
                period_me2=self.p.l,
                period_signal=self.p.m,
            )

        def next(self) -> None:
            if len(self.data.close) < 2:
                return
            dif = self.macd.macd
            dea = self.macd.signal
            crosses_down = dea[-1] < dif[-1] and dea[0] > dif[0]
            crosses_up = dif[-1] < dea[-1] and dif[0] > dea[0]
            if crosses_down:
                self.close()
            if crosses_up:
                self.buy()

    lite_value, lite_trades = _run_lite(lite_strategy)
    engine_value, engine_trades = _run_engine(EngineMACDCross)

    assert lite_value > 0
    assert lite_trades >= 0
    assert engine_value > 0
    assert engine_trades >= 0


def test_tradelearn_1x_strategy_examples_are_runnable() -> None:
    cases = [
        ("03_1x_sma_cross", "OneXSmaCross", {}),
        ("04_1x_bollinger", "OneXBollBandCross", {"title": "Long&Short"}),
        ("05_1x_macd", "OneXMACDCross", {"title": "Long&Short"}),
        ("06_1x_grid_trade", "OneXGridTrade", {}),
        ("07_1x_turtle", "OneXAdvancedTurtle", {"title": "Long"}),
    ]

    for module_name, class_name, params in cases:
        value, trades = _run_lite(_load_strategy(module_name, class_name), **params)
        assert value > 0
        assert trades >= 0
