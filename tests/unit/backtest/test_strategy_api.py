from __future__ import annotations

import pandas as pd
import pytest

import tradelearn
from tradelearn.backtest import Analyzer, Cerebro, Order, SimBroker, Strategy


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


def test_cerebro_runs_strategy_with_params_and_current_bar_index() -> None:
    class RecordingStrategy(Strategy):
        params = (("threshold", 10.5),)

        def __init__(self) -> None:
            self.seen_in_init = self.p.threshold
            self.values: list[tuple[float, float | None]] = []

        def next(self) -> None:
            previous = None if len(self.values) == 0 else self.data.close[-1]
            self.values.append((self.data.close[0], previous))

    cerebro = Cerebro()
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(RecordingStrategy, threshold=11.0)

    [strategy] = cerebro.run()

    assert strategy.seen_in_init == 11.0
    assert strategy.p.threshold == 11.0
    assert strategy.params.threshold == 11.0
    assert strategy.data is strategy.datas[0]
    assert strategy.data._name == "daily"
    assert strategy.values == [(10.0, None), (11.0, 10.0), (12.0, 11.0)]


def test_analyzer_receives_strategy_and_bar_lifecycle() -> None:
    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    class CloseAnalyzer(Analyzer):
        params = (("scale", 1.0),)

        def __init__(self) -> None:
            self.started = False
            self.values: list[float] = []
            self.ended = False

        def on_start(self) -> None:
            self.started = self.strategy is not None

        def on_bar(self, bar) -> None:
            self.values.append(bar.close * self.p.scale)

        def on_end(self, stats) -> None:
            self.ended = stats["bars"] == 3

        def get_analysis(self) -> dict[str, object]:
            return {"started": self.started, "values": self.values, "ended": self.ended}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(CloseAnalyzer, scale=2.0, name="close")

    [strategy] = cerebro.run()

    assert strategy.analyzers["close"].get_analysis() == {
        "started": True,
        "values": [20.0, 22.0, 24.0],
        "ended": True,
    }


def test_cerebro_exposes_named_analyzers_and_run_analysis() -> None:
    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    class CloseAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.values: list[float] = []

        def on_bar(self, bar) -> None:
            self.values.append(bar.close)

        def get_analysis(self) -> dict[str, object]:
            return {"values": self.values}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(CloseAnalyzer, name="close")

    [strategy] = cerebro.run()

    assert strategy.analyzers.close.get_analysis() == {"values": [10.0, 11.0, 12.0]}
    assert strategy.analyzers.getbyname("close") is strategy.analyzers.close
    assert strategy.analyzer_results == {"close": {"values": [10.0, 11.0, 12.0]}}
    assert cerebro.analyzer_results == {"close": {"values": [10.0, 11.0, 12.0]}}


def test_strategy_params_must_be_tuple_pairs() -> None:
    class BadParams(Strategy):
        params = {"fast": 10}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(BadParams)

    with pytest.raises(ValueError, match="Strategy.params must be a tuple"):
        cerebro.run()


def test_simbroker_executes_pending_orders_and_notifies_strategy_and_analyzers() -> None:
    class BuyThenClose(Strategy):
        def __init__(self) -> None:
            self.orders = []
            self.trades = []
            self.values: list[tuple[float, float]] = []

        def next(self) -> None:
            self.values.append((self.position.size, self.broker.getcash()))
            if len(self.values) == 1:
                self.buy(size=2)
            elif len(self.values) == 2:
                self.close()

        def notify_order(self, order) -> None:
            self.orders.append((order.status, order.ordtype, order.executed.price))

        def notify_trade(self, trade) -> None:
            self.trades.append((trade.size, trade.price, trade.isopen, trade.isclosed))

    class FillAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.fills = []
            self.trades = []

        def on_fill(self, fill) -> None:
            self.fills.append((fill.size, fill.price))

        def on_trade(self, trade) -> None:
            self.trades.append((trade.size, trade.price, trade.isopen, trade.isclosed))

        def get_analysis(self) -> dict[str, object]:
            return {"fills": self.fills, "trades": self.trades}

    cerebro = Cerebro()
    assert isinstance(cerebro.broker, SimBroker)
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyThenClose)
    cerebro.addanalyzer(FillAnalyzer, name="fills")

    [strategy] = cerebro.run()

    assert strategy.values == [(0.0, 100.0), (2.0, 80.0), (0.0, 102.0)]
    assert strategy.orders == [
        (Order.Accepted, Order.Buy, 0.0),
        (Order.Completed, Order.Buy, 10.0),
        (Order.Accepted, Order.Sell, 0.0),
        (Order.Completed, Order.Sell, 11.0),
    ]
    assert strategy.trades == [
        (2.0, 10.0, True, False),
        (0.0, 11.0, False, True),
    ]
    assert strategy.position.size == 0.0
    assert strategy.position.price == 0.0
    assert strategy.broker.getcash() == 102.0
    assert strategy.broker.getvalue() == 102.0
    assert strategy.analyzers["fills"].get_analysis() == {
        "fills": [(2.0, 10.0), (-2.0, 11.0)],
        "trades": [(2.0, 10.0, True, False), (0.0, 11.0, False, True)],
    }


def test_simbroker_getvalue_marks_open_position_to_current_close() -> None:
    class BuyAndHold(Strategy):
        def __init__(self) -> None:
            self.values: list[float] = []

        def next(self) -> None:
            self.values.append(self.broker.getvalue())
            if not self.position:
                self.buy(size=2)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyAndHold)

    [strategy] = cerebro.run()

    assert strategy.values == [100.0, 102.0, 104.0]
    assert strategy.broker.getcash() == 80.0
    assert strategy.broker.getvalue() == 104.0


def test_simbroker_limit_order_waits_until_bar_crosses_limit() -> None:
    class BuyLimitBelowMarket(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []
            self.cash_values: list[float] = []

        def next(self) -> None:
            self.cash_values.append(self.broker.getcash())
            if len(self.cash_values) == 1:
                self.buy(size=2, price=8.5, exectype=Order.Limit)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyLimitBelowMarket)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Accepted]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 100.0
    assert strategy.cash_values == [100.0, 100.0, 100.0]


def test_simbroker_prefers_rust_match_order_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class RustBridge:
        @staticmethod
        def match_order_fill(
            order_id,
            symbol,
            side,
            order_type,
            size,
            limit_price,
            stop_price,
            created_ts,
            ts,
            open_,
            high,
            low,
            close,
            volume,
            trade_on_close,
            commission_ratio,
        ):
            calls.append(
                {
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "order_type": order_type,
                    "size": size,
                    "open": open_,
                    "commission_ratio": commission_ratio,
                }
            )
            return (size, open_ + 0.5, 0.0, 0.5)

    class BuyOnce(Strategy):
        def __init__(self) -> None:
            self.order_prices: list[float] = []

        def next(self) -> None:
            if not self.position:
                self.buy(size=2)

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.order_prices.append(order.executed.price)

    monkeypatch.setattr(tradelearn, "_rust", RustBridge(), raising=False)
    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.broker.setcommission(0.01)
    cerebro.adddata(bars(), name="bridge")
    cerebro.addstrategy(BuyOnce)

    [strategy] = cerebro.run()

    assert calls == [
        {
            "order_id": 1,
            "symbol": "bridge",
            "side": "buy",
            "order_type": "market",
            "size": 2.0,
            "open": 10.0,
            "commission_ratio": 0.01,
        }
    ]
    assert strategy.order_prices == [10.5]
    assert strategy.broker.getcash() == 79.0
