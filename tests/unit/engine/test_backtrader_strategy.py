from __future__ import annotations

import logging

import pandas as pd

import tradelearn.engine as bt
from tradelearn.engine import Cerebro


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


def test_backtrader_strategy_import_path_preserves_params_and_line_indexing() -> None:
    class RecordingStrategy(bt.Strategy):
        params = (("threshold", 10.5),)

        def __init__(self) -> None:
            self.threshold_seen = self.p.threshold
            self.values: list[tuple[float, float | None]] = []

        def next(self) -> None:
            previous = None if len(self.values) == 0 else self.data.close[-1]
            self.values.append((self.data.close[0], previous))

    cerebro = Cerebro()
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)

    [strategy] = cerebro.run()

    assert strategy.threshold_seen == 11.0
    assert strategy.p.threshold == 11.0
    assert strategy.data._name == "daily"
    assert strategy.values == [(10.0, None), (11.0, 10.0), (12.0, 11.0)]


def test_backtrader_strategy_history_panel_returns_multi_data_window() -> None:
    data_b = bars().assign(
        open=[19.0, 20.0, 21.0],
        high=[21.0, 22.0, 23.0],
        low=[18.0, 19.0, 20.0],
        close=[20.0, 21.0, 22.0],
    )

    class RecordingStrategy(bt.Strategy):
        history = None

        def next(self) -> None:
            if len(self.data) == 3:
                self.history = self.history_panel(lookback=2)

    cerebro = Cerebro()
    cerebro.adddata(bars(), name="AAA")
    cerebro.adddata(data_b, name="BBB")
    cerebro.addstrategy(RecordingStrategy)

    [strategy] = cerebro.run()
    history = strategy.history

    assert history.index.names == ["timestamp", "symbol"]
    assert list(history.index.get_level_values("symbol").unique()) == ["AAA", "BBB"]
    assert list(history.index.get_level_values("timestamp").unique()) == list(
        pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    )
    assert history.loc[(pd.Timestamp("2026-01-03", tz="UTC"), "BBB"), "close"] == 22.0


def test_engine_cerebro_logs_run_start_and_summary(caplog) -> None:
    class LoggingStrategy(bt.Strategy):
        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(size=1)

    caplog.set_level(logging.INFO, logger="tradelearn.engine.cerebro")
    cerebro = Cerebro()
    cerebro.setcash(1000.0)
    cerebro.setcommission(0.001)
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(LoggingStrategy)

    cerebro.run()

    messages = [record.getMessage() for record in caplog.records]
    assert any("Cerebro run started" in message and "LoggingStrategy" in message for message in messages)
    assert any(
        "Cerebro run finished" in message
        and "final_value=" in message
        and "return_pct=" in message
        for message in messages
    )


def test_backtrader_strategy_module_exports_order_and_line_types() -> None:
    line = bt.LineSeries([1.0, 2.0, 3.0])
    line._advance(1)

    assert line[0] == 2.0
    assert line[-1] == 1.0
    assert bt.Order.Completed == 4


def test_engine_ignores_orders_created_on_terminal_bar() -> None:
    class TerminalOrderStrategy(bt.Strategy):
        def next(self) -> None:
            if len(self.data) == 3:
                self.buy(size=1)

    cerebro = Cerebro(trade_on_close=True)
    cerebro.setcash(1000.0)
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(TerminalOrderStrategy)

    [strategy] = cerebro.run()

    assert strategy.stats.summary["final_value"] == 1000.0
    assert strategy.stats.orders.empty
    assert strategy.stats.fills.empty
