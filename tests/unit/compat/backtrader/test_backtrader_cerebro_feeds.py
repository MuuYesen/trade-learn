from __future__ import annotations

import pandas as pd

import tradelearn.compat.backtrader as bt
from tradelearn.backtest import Analyzer


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

    cerebro = bt.Cerebro(trade_on_close=True)
    data = bt.feeds.PandasData(dataname=bars(), name="daily")
    returned = cerebro.adddata(data)
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)
    cerebro.addanalyzer(CloseAnalyzer, name="close")

    [strategy] = cerebro.run()

    assert returned is data
    assert strategy.data._name == "daily"
    assert strategy.values == [10.0, 11.0, 12.0]
    assert strategy.analyzer_results == {"close": {"values": [10.0, 11.0, 12.0]}}
    assert strategy.stats.fills[["size", "price"]].to_dict("records") == [
        {"size": 1.0, "price": 11.0}
    ]


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
