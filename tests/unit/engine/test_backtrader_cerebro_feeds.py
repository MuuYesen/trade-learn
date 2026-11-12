from __future__ import annotations

from tests.unit.backtest._engine_config import configured_engine

import pandas as pd
import pytest

import tradelearn.engine as bt
from tradelearn.backtest import engine
from tradelearn.engine import Analyzer


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [9.0, 10.0, 11.0],
            "high": [11.0, 12.0, 13.0],
            "low": [8.0, 9.0, 10.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )


def panel_bars() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, offset in (("AAA", 0.0), ("BBB", 100.0)):
        frame = bars().copy()
        frame["symbol"] = symbol
        frame["close"] = frame["close"] + offset
        frame["open"] = frame["open"] + offset
        frame["high"] = frame["high"] + offset
        frame["low"] = frame["low"] + offset
        frames.append(frame.reset_index(names="timestamp"))
    return (
        pd.concat(frames, ignore_index=True)
        .set_index(["timestamp", "symbol"])
        .sort_index()
    )


def test_backtrader_cerebro_runs_pandasdata_feed_and_analyzer() -> None:
    class RecordingStrategy(bt.Strategy):
        params = (("threshold", 10.5),)

        def __init__(self) -> None:
            self.values: list[float] = []

        def next(self) -> None:
            self.values.append(self.data.close[0])
            if not self.position and self.data.close[0] >= self.p.threshold:
                self.buy(size=1)

    class CloseAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.values: list[float] = []

        def on_bar(self, bar) -> None:
            self.values.append(bar.close)

        def get_analysis(self) -> dict[str, list[float]]:
            return {"values": self.values}

    cerebro = configured_engine(trade_on_close=True)
    data = bt.feeds.PandasData(dataname=bars(), name="daily")
    returned = cerebro.adddata(data)
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)
    cerebro.addanalyzer(CloseAnalyzer, _name="close")

    [strategy] = cerebro.run()

    assert returned is data
    assert strategy.data._name == "daily"
    assert strategy.values == [10.0, 11.0, 12.0]
    assert strategy.analyzer_results == {"close": {"values": [10.0, 11.0, 12.0]}}
    assert strategy.stats.fills[["size", "price"]].to_dict("records") == [
        {"size": 1.0, "price": 11.0}
    ]


def test_non_streaming_analyzer_does_not_build_bar_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoopStrategy(bt.Strategy):
        def next(self) -> None:
            pass

    class SummaryAnalyzer(Analyzer):
        metric_key = "returns"

    def fail_current_bar(data):
        raise AssertionError("bar snapshots should be lazy for non-streaming analyzers")

    monkeypatch.setattr(engine, "_current_bar", fail_current_bar)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=bars(), name="daily"))
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(SummaryAnalyzer, _name="summary")

    [strategy] = cerebro.run()

    assert "summary" in strategy.analyzer_results


def test_pandasdata_accepts_backtrader_style_column_mapping() -> None:
    frame = bars().rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    data = bt.feeds.PandasData(
        dataname=frame,
        name="mapped",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    data._advance(0)

    assert data._name == "mapped"
    assert data.open[0] == 9.0
    assert data.close[0] == 10.0


def test_cerebro_adddata_requires_explicit_feed_for_panels():
    c=bt.Cerebro()
    with pytest.raises(TypeError):c.adddata(panel_bars())
    for symbol in ('AAA','BBB'):
        c.adddata(bt.feeds.PandasData(dataname=panel_bars().xs(symbol,level='symbol')),name=symbol)
    assert list(c.datasbyname)==['AAA','BBB']


@pytest.mark.parametrize("unsupported", [{"typo_column": "close"}])
def test_pandasdata_rejects_unimplemented_parameters(unsupported):
    with pytest.raises(TypeError):
        bt.feeds.PandasData(dataname=bars(), **unsupported)
