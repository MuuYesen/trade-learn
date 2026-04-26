from __future__ import annotations

import pandas as pd
import pytest

import tradelearn
from tradelearn.backtest import Analyzer, Cerebro, Order, SimBroker, Stats, Strategy


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
            self.ended = stats.summary["bars"] == 3

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


def test_cerebro_aligns_secondary_data_to_primary_clock() -> None:
    primary = bars()
    secondary = pd.DataFrame(
        {
            "open": [90.0, 120.0],
            "high": [91.0, 121.0],
            "low": [89.0, 119.0],
            "close": [100.0, 130.0],
            "volume": [500.0, 700.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True),
    )

    class MultiDataStrategy(Strategy):
        def __init__(self) -> None:
            self.rows: list[tuple[pd.Timestamp, float, pd.Timestamp, float]] = []

        def next(self) -> None:
            self.rows.append(
                (
                    self.data.datetime[0],
                    self.data.close[0],
                    self.datas[1].datetime[0],
                    self.datas[1].close[0],
                )
            )

    cerebro = Cerebro()
    cerebro.adddata(primary, name="daily")
    cerebro.adddata(secondary, name="sparse")
    cerebro.addstrategy(MultiDataStrategy)

    [strategy] = cerebro.run()

    assert strategy.rows == [
        (
            pd.Timestamp("2026-01-01", tz="UTC"),
            10.0,
            pd.Timestamp("2026-01-01", tz="UTC"),
            100.0,
        ),
        (
            pd.Timestamp("2026-01-02", tz="UTC"),
            11.0,
            pd.Timestamp("2026-01-01", tz="UTC"),
            100.0,
        ),
        (
            pd.Timestamp("2026-01-03", tz="UTC"),
            12.0,
            pd.Timestamp("2026-01-03", tz="UTC"),
            130.0,
        ),
    ]


def test_strategy_addminperiod_skips_next_until_warmup_finishes() -> None:
    class WarmupStrategy(Strategy):
        def __init__(self) -> None:
            self.addminperiod(2)
            self.prenext_values: list[float] = []
            self.next_values: list[float] = []

        def prenext(self) -> None:
            self.prenext_values.append(self.data.close[0])

        def next(self) -> None:
            self.next_values.append(self.data.close[0])

    class CloseAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.values: list[float] = []

        def on_bar(self, bar) -> None:
            self.values.append(bar.close)

        def get_analysis(self) -> dict[str, object]:
            return {"values": self.values}

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(WarmupStrategy)
    cerebro.addanalyzer(CloseAnalyzer, name="close")

    [strategy] = cerebro.run()

    assert strategy.prenext_values == [10.0, 11.0]
    assert strategy.next_values == [12.0]
    assert strategy.analyzers.close.get_analysis() == {"values": [10.0, 11.0, 12.0]}


def test_strategy_min_period_class_attribute_controls_warmup() -> None:
    class WarmupStrategy(Strategy):
        min_period = 1

        def __init__(self) -> None:
            self.prenext_values: list[float] = []
            self.next_values: list[float] = []

        def prenext(self) -> None:
            self.prenext_values.append(self.data.close[0])

        def next(self) -> None:
            self.next_values.append(self.data.close[0])

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(WarmupStrategy)

    [strategy] = cerebro.run()

    assert strategy.prenext_values == [10.0]
    assert strategy.next_values == [11.0, 12.0]


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
        (Order.Submitted, Order.Buy, 0.0),
        (Order.Accepted, Order.Buy, 0.0),
        (Order.Completed, Order.Buy, 10.0),
        (Order.Submitted, Order.Sell, 0.0),
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


def test_cerebro_exposes_report_ready_stats_artifacts() -> None:
    class BuyThenClose(Strategy):
        def __init__(self) -> None:
            self.values: list[tuple[float, float]] = []

        def next(self) -> None:
            self.values.append((self.position.size, self.broker.getcash()))
            if len(self.values) == 1:
                self.buy(size=2)
            elif len(self.values) == 2:
                self.close()

    class FillAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.fill_count = 0

        def on_fill(self, fill) -> None:
            self.fill_count += 1

        def get_analysis(self) -> dict[str, object]:
            return {"fill_count": self.fill_count}

    cerebro = Cerebro(callback_batch=3, exactbars=True, stdstats=False)
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(BuyThenClose)
    cerebro.addanalyzer(FillAnalyzer, name="fills")

    [strategy] = cerebro.run()

    assert isinstance(cerebro.stats, Stats)
    assert strategy.stats is cerebro.stats
    assert strategy.analyzer_results == {"fills": {"fill_count": 2}}
    pd.testing.assert_series_equal(
        strategy.stats.equity,
        pd.Series(
            [100.0, 102.0, 102.0],
            index=bars().index,
            name="equity",
        ),
    )
    pd.testing.assert_series_equal(
        strategy.stats.returns,
        pd.Series(
            [0.0, 0.02, 0.0],
            index=bars().index,
            name="returns",
        ),
    )
    assert set(
        [
            "ref",
            "datetime",
            "data",
            "side",
            "exectype",
            "status",
            "size",
            "executed_size",
            "executed_price",
        ]
    ).issubset(strategy.stats.orders.columns)
    assert strategy.stats.orders["status"].tolist() == [
        "Submitted",
        "Accepted",
        "Completed",
        "Submitted",
        "Accepted",
        "Completed",
    ]
    assert strategy.stats.fills[["order_ref", "data", "size", "price"]].to_dict("records") == [
        {"order_ref": 1, "data": "daily", "size": 2.0, "price": 10.0},
        {"order_ref": 2, "data": "daily", "size": -2.0, "price": 11.0},
    ]
    assert strategy.stats.trades[["ref", "data", "size", "price", "pnl"]].to_dict("records") == [
        {"ref": 1, "data": "daily", "size": 2.0, "price": 10.0, "pnl": 0.0},
        {"ref": 2, "data": "daily", "size": 0.0, "price": 11.0, "pnl": 2.0},
    ]
    assert strategy.stats.positions[["datetime", "data", "size", "mark_price"]].to_dict(
        "records"
    ) == [
        {
            "datetime": pd.Timestamp("2026-01-02", tz="UTC"),
            "data": "daily",
            "size": 2.0,
            "mark_price": 11.0,
        },
        {
            "datetime": pd.Timestamp("2026-01-03", tz="UTC"),
            "data": "daily",
            "size": 0.0,
            "mark_price": 12.0,
        },
    ]
    assert strategy.stats.summary == {
        "bars": 3,
        "final_cash": 102.0,
        "final_value": 102.0,
        "final_realized_pnl": 2.0,
        "final_unrealized_pnl": 0.0,
        "final_margin_used": 0.0,
        "total_trades": 2,
        "total_orders": 6,
        "total_fills": 2,
    }
    assert strategy.stats.analyzers == {"fills": {"fill_count": 2}}
    assert strategy.stats.config == {
        "callback_batch": 3,
        "trade_on_close": False,
        "exactbars": True,
        "stdstats": False,
        "broker": {"cash": 102.0, "commission": 0.0},
    }


def test_positions_artifact_tracks_flat_snapshot_realized_pnl_and_margin() -> None:
    class BuyThenClose(Strategy):
        def __init__(self) -> None:
            self.seen = 0

        def next(self) -> None:
            self.seen += 1
            if self.seen == 1:
                self.buy(size=2)
            elif self.seen == 2:
                self.close()

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars(), name="daily")
    cerebro.addstrategy(BuyThenClose)

    [strategy] = cerebro.run()

    assert strategy.stats.positions[
        [
            "datetime",
            "data",
            "size",
            "avg_price",
            "mark_price",
            "value",
            "unrealized_pnl",
            "realized_pnl",
            "margin_used",
        ]
    ].to_dict("records") == [
        {
            "datetime": pd.Timestamp("2026-01-02", tz="UTC"),
            "data": "daily",
            "size": 2.0,
            "avg_price": 10.0,
            "mark_price": 11.0,
            "value": 22.0,
            "unrealized_pnl": 2.0,
            "realized_pnl": 0.0,
            "margin_used": 22.0,
        },
        {
            "datetime": pd.Timestamp("2026-01-03", tz="UTC"),
            "data": "daily",
            "size": 0.0,
            "avg_price": 0.0,
            "mark_price": 12.0,
            "value": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 2.0,
            "margin_used": 0.0,
        },
    ]
    assert strategy.stats.summary["final_realized_pnl"] == 2.0
    assert strategy.stats.summary["final_unrealized_pnl"] == 0.0
    assert strategy.stats.summary["final_margin_used"] == 0.0


def test_analyzer_on_end_receives_stats_object() -> None:
    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    class StatsAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.stats_type = ""
            self.final_value = 0.0
            self.equity_points = 0

        def on_end(self, stats) -> None:
            self.stats_type = type(stats).__name__
            self.final_value = stats.summary["final_value"]
            self.equity_points = len(stats.equity)

        def get_analysis(self) -> dict[str, object]:
            return {
                "stats_type": self.stats_type,
                "final_value": self.final_value,
                "equity_points": self.equity_points,
            }

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(StatsAnalyzer, name="stats")

    [strategy] = cerebro.run()

    assert strategy.analyzer_results == {
        "stats": {"stats_type": "Stats", "final_value": 100.0, "equity_points": 3}
    }
    assert strategy.stats.analyzers == strategy.analyzer_results


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


def test_trade_on_close_executes_new_market_orders_on_current_close() -> None:
    data = pd.DataFrame(
        {
            "open": [9.0, 20.0, 21.0],
            "high": [11.0, 22.0, 23.0],
            "low": [8.0, 19.0, 20.0],
            "close": [10.0, 21.0, 22.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
    )

    class BuyOnClose(Strategy):
        def __init__(self) -> None:
            self.values: list[tuple[float, float]] = []
            self.order_prices: list[float] = []

        def next(self) -> None:
            self.values.append((self.position.size, self.broker.getcash()))
            if not self.position:
                self.buy(size=2)

        def notify_order(self, order) -> None:
            if order.status == Order.Completed:
                self.order_prices.append(order.executed.price)

    cerebro = Cerebro(trade_on_close=True)
    cerebro.broker.setcash(100.0)
    cerebro.adddata(data)
    cerebro.addstrategy(BuyOnClose)

    [strategy] = cerebro.run()

    assert strategy.order_prices == [10.0]
    assert strategy.values == [(0.0, 100.0), (2.0, 80.0), (2.0, 80.0)]
    assert strategy.position.size == 2.0
    assert strategy.broker.getcash() == 80.0
    assert strategy.broker.getvalue() == 124.0


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

    assert strategy.statuses == [Order.Submitted, Order.Accepted]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 100.0
    assert strategy.cash_values == [100.0, 100.0, 100.0]


def test_simbroker_limit_order_fills_after_bar_crosses_limit() -> None:
    class BuyLimitAtMarket(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.position and not self.statuses:
                self.buy(size=2, price=9.5, exectype=Order.Limit)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyLimitAtMarket)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Completed]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 9.5
    assert strategy.broker.getcash() == 81.0


def test_simbroker_stop_order_triggers_on_bar_high() -> None:
    class BuyStop(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.position and not self.statuses:
                self.buy(size=2, price=11.5, exectype=Order.Stop)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyStop)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Completed]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 10.0
    assert strategy.broker.getcash() == 80.0


def test_simbroker_stop_limit_uses_distinct_stop_and_limit_prices() -> None:
    class BuyStopLimit(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.position and not self.statuses:
                self.buy(size=2, price=11.5, pricelimit=10.5, exectype=Order.StopLimit)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyStopLimit)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Completed]
    assert strategy.position.size == 2.0
    assert strategy.position.price == 10.0
    assert strategy.broker.getcash() == 80.0


def test_simbroker_ioc_order_cancels_when_not_filled_immediately() -> None:
    class BuyLimitIoc(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.statuses:
                self.buy(size=2, price=8.5, exectype=Order.Limit, time_in_force=Order.IOC)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyLimitIoc)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Canceled]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 100.0


def test_analyzer_receives_order_lifecycle_events_for_cancel_and_reject() -> None:
    class CancelAndReject(Strategy):
        def __init__(self) -> None:
            self.submitted = False

        def next(self) -> None:
            if not self.submitted:
                self.buy(size=2, price=8.5, exectype=Order.Limit, time_in_force=Order.IOC)
                self.buy(size=2000)
                self.submitted = True

    class OrderAnalyzer(Analyzer):
        def __init__(self) -> None:
            self.events: list[tuple[int, int, int]] = []

        def on_order(self, order) -> None:
            self.events.append((order.ref, order.ordtype, order.status))

        def get_analysis(self) -> dict[str, object]:
            return {"events": self.events}

    cerebro = Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(CancelAndReject)
    cerebro.addanalyzer(OrderAnalyzer, name="orders")

    [strategy] = cerebro.run()

    assert strategy.analyzers.orders.get_analysis() == {
        "events": [
            (1, Order.Buy, Order.Submitted),
            (1, Order.Buy, Order.Accepted),
            (2, Order.Buy, Order.Submitted),
            (2, Order.Buy, Order.Accepted),
            (1, Order.Buy, Order.Canceled),
            (2, Order.Buy, Order.Rejected),
        ]
    }


def test_simbroker_day_order_expires_after_first_unfilled_bar() -> None:
    class BuyLimitDay(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.statuses:
                self.buy(size=2, price=8.5, exectype=Order.Limit, time_in_force=Order.DAY)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(BuyLimitDay)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Expired]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 100.0


def test_simbroker_rejects_partial_fill_when_bar_volume_is_insufficient() -> None:
    class OversizedBuy(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.statuses:
                self.buy(size=2000)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(OversizedBuy)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Rejected]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 100000.0


def test_simbroker_rejects_buy_when_cash_is_insufficient() -> None:
    class UnderfundedBuy(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.statuses:
                self.buy(size=2)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(10.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(UnderfundedBuy)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Rejected]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 10.0
    assert strategy.broker.getvalue() == 10.0


def test_simbroker_rejects_short_when_margin_cash_is_insufficient() -> None:
    class UnderfundedShort(Strategy):
        def __init__(self) -> None:
            self.statuses: list[int] = []

        def next(self) -> None:
            if not self.statuses:
                self.sell(size=2)

        def notify_order(self, order) -> None:
            self.statuses.append(order.status)

    cerebro = Cerebro()
    cerebro.broker.setcash(10.0)
    cerebro.adddata(bars())
    cerebro.addstrategy(UnderfundedShort)

    [strategy] = cerebro.run()

    assert strategy.statuses == [Order.Submitted, Order.Accepted, Order.Rejected]
    assert strategy.position.size == 0.0
    assert strategy.broker.getcash() == 10.0
    assert strategy.broker.margin_used() == 0.0


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
