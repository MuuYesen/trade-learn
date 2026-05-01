from __future__ import annotations

from pathlib import Path

import pandas as pd

import tradelearn as tl
import tradelearn.engine as bt
from tradelearn.lite import Backtest, MLStrategy, Strategy
from tradelearn.lite.backtest import _can_skip_normalize_data


def _data() -> pd.DataFrame:
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


def test_lite_backtest_detects_normalized_data_for_feed_fast_path() -> None:
    assert _can_skip_normalize_data(_data())
    assert not _can_skip_normalize_data(_data().rename(columns={"open": "Open"}))


def test_lite_target_weight_snapshots_batch_prices_and_positions() -> None:
    seen: dict[str, dict[str, float]] = {}
    panel = {
        "AAA": _data(),
        "BBB": _data().assign(
            open=[20.0, 21.0, 22.0, 23.0, 24.0],
            high=[21.0, 22.0, 23.0, 24.0, 25.0],
            low=[19.0, 20.0, 21.0, 22.0, 23.0],
            close=[20.0, 21.0, 22.0, 23.0, 24.0],
        ),
    }

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                snapshots = self._target_weight_snapshots(self._target_weight_data_map())
                seen["prices"] = snapshots.prices
                seen["sizes"] = snapshots.sizes
                seen["mults"] = snapshots.mults

    Backtest(panel, LiteStrategy, cash=1000.0).run()

    assert seen == {
        "prices": {"AAA": 11.0, "BBB": 21.0},
        "sizes": {"AAA": 0.0, "BBB": 0.0},
        "mults": {"AAA": 1.0, "BBB": 1.0},
    }


def test_lite_uses_backtrader_bar_indexing_with_lite_position_call() -> None:
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.line = self.I(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=self.data.index))
            self.sma = tl.talib.SMA(self.data.close, timeperiod=2)
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                seen["close_now"] = self.data.close[0]
                seen["close_prev"] = self.data.close[-1]
                seen["line_now"] = self.line[0]
                seen["line_prev"] = self.line[-1]
                seen["sma_now"] = self.sma[0]
                assert not self.position()
                self.buy(size=1)
            elif len(self.data) == 4:
                assert self.position()
                self.position().close()

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert stats["total_trades"] == 1
    assert seen == {
        "close_now": 12.0,
        "close_prev": 11.0,
        "line_now": 3.0,
        "line_prev": 2.0,
        "sma_now": 11.5,
    }


def test_lite_run_exposes_shared_stats_summary_keys() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                self.buy(size=1)
            elif len(self.data) == 4:
                self.position().close()

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()
    shared_stats = stats.strategy.stats

    assert stats.stats is shared_stats
    assert stats.summary == shared_stats.summary
    assert stats.equity is shared_stats.equity
    assert stats.returns is shared_stats.returns
    assert stats.fills is shared_stats.fills
    assert stats.trades is shared_stats.trades
    assert stats.positions is shared_stats.positions
    assert stats.orders is shared_stats.orders
    assert stats.config is shared_stats.config
    assert stats.records == {}
    for key, value in shared_stats.summary.items():
        assert key in stats
        if value == value:
            assert stats[key] == value
    assert "return_pct" in stats
    assert "win_rate_pct" in stats
    assert "Equity Final [$]" not in stats
    assert "Return [%]" not in stats
    assert "# Trades" not in stats
    assert "Win Rate [%]" not in stats
    assert "_stats" not in stats
    assert "_strategy" not in stats
    assert "_records" not in stats


def test_lite_fractional_size_is_order_quantity_not_equity_fraction() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                self.buy(size=0.5)

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert stats.orders.iloc[0]["size"] == 0.5
    assert stats.fills.iloc[0]["size"] == 0.5


def test_lite_and_engine_stats_summary_keys_match() -> None:
    class LiteStrategy(Strategy):
        def next(self) -> None:
            if len(self.data) == 3:
                self.buy(size=1)
            elif len(self.data) == 4:
                self.position().close()

    class EngineStrategy(bt.Strategy):
        def next(self) -> None:
            if len(self.data) == 3:
                self.buy(size=1)
            elif len(self.data) == 4:
                self.close()

    lite_stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000.0)
    cerebro.adddata(bt.feeds.PandasData(dataname=_data(), name="data"))
    cerebro.addstrategy(EngineStrategy)
    [engine_strategy] = cerebro.run()

    assert set(lite_stats) == set(engine_strategy.stats.summary)


def test_lite_data_ta_returns_current_bar_line_proxy() -> None:
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.atr = tl.talib.ATR(
                self.data.high,
                self.data.low,
                self.data.close,
                timeperiod=2,
            )
            self.macd = tl.talib.MACD(
                self.data.close,
                fastperiod=2,
                slowperiod=3,
                signalperiod=2,
            )
            self.start_on_bar(3)

        def next(self) -> None:
            seen["atr_now"] = self.atr[0]
            seen["atr_prev"] = self.atr[-1]
            seen["macd_now"] = self.macd.macd[0]
            seen["macd_prev"] = self.macd.macd[-1]

    Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert seen["atr_now"] == 2.0
    assert seen["atr_prev"] == 2.0
    assert "macd_now" in seen
    assert "macd_prev" in seen


def test_lite_i_accepts_dataframe_indicator_and_slice_columns() -> None:
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            frame = pd.DataFrame(
                {
                    "DIF": [0.0, 0.1, 0.3, 0.2, 0.4],
                    "DEA": [0.0, 0.0, 0.2, 0.25, 0.3],
                },
                index=self.data.index,
            )
            self.macd = self.I(frame, name="macd")
            self.start_on_bar(2)

        def next(self) -> None:
            dif = self.macd[:, 0]
            dea = self.macd[:, 1]
            seen["dif_now"] = dif[0]
            seen["dea_prev"] = dea[-1]

    Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert seen == {"dif_now": 0.4, "dea_prev": 0.25}


def test_lite_records_are_exposed_from_run_result() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            self.record(signal=self.data.close[0])

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    records = stats.records
    assert records["signal"].dropna().tolist() == [12.0, 13.0, 14.0]


def test_lite_does_not_export_trading_signal_api() -> None:
    assert "Signal" not in tl.lite.__all__
    assert "SignalStrategy" not in tl.lite.__all__
    assert not hasattr(tl.lite, "Signal")
    assert not hasattr(tl.lite, "SignalStrategy")


def test_lite_rejects_sl_tp_until_bracket_orders_are_implemented() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(size=1, sl=9.0, tp=13.0)

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()
    orders = stats.strategy.orders

    assert len(orders) == 3
    assert orders[1].parent is orders[0]
    assert orders[2].oco is orders[1]


def test_lite_exposes_engine_order_target_helpers() -> None:
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.order_target_size(target=3)
            elif len(self.data) == 3:
                seen["size_after_target"] = self.position().size
                self.order_target_value(target=24.0)
            elif len(self.data) == 4:
                seen["size_after_value"] = self.position().size
                self.order_target_percent(target=0.0)

    Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert seen["size_after_target"] == 3
    assert seen["size_after_value"] == 2


def test_lite_supports_explicit_bracket_helpers() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                orders = self.buy_bracket(size=1, sl=9.0, tp=13.0)
                assert len(orders) == 3
                assert orders[1].parent is orders[0]
                assert orders[2].oco is orders[1]

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert len(stats.strategy.orders) == 3


def test_lite_exports_mlstrategy_with_shared_ml_runtime() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.fit_X = None
            self.fit_y = None
            self.predict_X = []

        def fit(self, X, y):
            self.fit_X = X
            self.fit_y = y
            return self

        def predict(self, X):
            self.predict_X.extend(X)
            return [1.0]

    class LiteML(MLStrategy):
        model = RecordingModel()
        features = ("close",)

        @staticmethod
        def target(data: pd.DataFrame) -> pd.Series:
            return (data["close"].shift(-1) > data["close"]).astype(float)

    stats = Backtest(_data(), LiteML, cash=1000.0, trade_on_close=True).run()
    strategy = stats.strategy

    assert strategy.model_.fit_X == [[10.0], [11.0], [12.0], [13.0], [14.0]]
    assert strategy.model_.fit_y == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert strategy.model_.predict_X == [[10.0], [11.0], [12.0], [13.0], [14.0]]
    assert len(strategy.orders) == 1
    assert strategy.orders[0].isbuy()


def test_lite_accepts_dict_data_and_ticker_orders() -> None:
    data = {
        "AAA": _data(),
        "BBB": _data().assign(close=[20.0, 21.0, 22.0, 23.0, 24.0]),
    }
    seen: dict[str, float] = {}

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(ticker="BBB", size=2)
            elif len(self.data) == 3:
                seen["bbb_size"] = self.position("BBB").size

    Backtest(data, LiteStrategy, cash=1000.0).run()

    assert seen["bbb_size"] == 2


def test_lite_target_weights_batches_equity_and_orders_reductions_first() -> None:
    data = {
        "AAA": _data().assign(close=[10.0, 10.0, 10.0, 10.0, 10.0]),
        "BBB": _data().assign(close=[20.0, 20.0, 20.0, 20.0, 20.0]),
        "CCC": _data().assign(close=[30.0, 30.0, 30.0, 30.0, 30.0]),
    }
    seen: dict[str, object] = {}

    class PortfolioLite(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(ticker="CCC", size=3)
            elif len(self.data) == 3:
                original_getvalue = self.broker.getvalue
                calls = {"count": 0}

                def counted_getvalue(*args, **kwargs):
                    calls["count"] += 1
                    return original_getvalue(*args, **kwargs)

                self.broker.getvalue = counted_getvalue
                before = len(self.orders)
                self.target_weights({"AAA": 0.2, "BBB": 0.2}, close_missing=True)
                new_orders = self.orders[before:]
                seen["getvalue_calls"] = calls["count"]
                seen["sides"] = ["buy" if order.isbuy() else "sell" for order in new_orders]
                seen["tickers"] = [getattr(order.data, "_name", None) for order in new_orders]

    Backtest(data, PortfolioLite, cash=1000.0).run()

    assert seen["getvalue_calls"] == 1
    assert seen["sides"][0] == "sell"
    assert seen["tickers"][0] == "CCC"


def test_lite_target_weights_targets_multi_asset_weights() -> None:
    data = {
        "AAA": _data(),
        "BBB": _data().assign(close=[20.0, 21.0, 22.0, 23.0, 24.0]),
    }

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.target_weights({"AAA": 0.25, "BBB": 0.50, "cash": 0.25})

    stats = Backtest(data, LiteStrategy, cash=1000.0, trade_on_close=True).run()
    orders = stats.strategy.orders

    assert [(order.data._name, order.size) for order in orders] == [("AAA", 22.0), ("BBB", 23.0)]


def test_lite_target_equal_and_close_all_are_target_position_sugar() -> None:
    data = {
        "AAA": _data(),
        "BBB": _data().assign(close=[20.0, 21.0, 22.0, 23.0, 24.0]),
    }

    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.target_equal(["AAA", "BBB"], weight=0.8)
            elif len(self.data) == 3:
                self.close_all()

    stats = Backtest(data, LiteStrategy, cash=1000.0, trade_on_close=True).run()

    submitted_orders = [
        (order.data._name, order.isbuy(), order.size)
        for order in stats.strategy.orders
    ]
    assert submitted_orders == [
        ("AAA", True, 36.0),
        ("BBB", True, 19.0),
        ("AAA", False, 36.0),
        ("BBB", False, 19.0),
    ]


def test_lite_strategy_does_not_expose_rebalance_or_alloc_lifecycle() -> None:
    assert "rebalance" not in Strategy.__dict__
    assert "alloc" not in Strategy.__dict__


def test_lite_examples_do_not_use_backtesting_py_surface() -> None:
    root = Path(__file__).parents[3]
    checked_paths = [root / "examples" / "lite"]
    forbidden = [
        "from backtesting import",
        "self.data.Close",
        "self.data.Open",
        "self.data.High",
        "self.data.Low",
        "self.data.Volume",
        "self.position.close()",
    ]
    offenders: list[str] = []
    for checked_path in checked_paths:
        for path in checked_path.glob("*.py"):
            text = path.read_text()
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_lite_strategy_module_stays_thin_facade() -> None:
    root = Path(__file__).parents[3]
    text = (root / "tradelearn" / "lite" / "strategy.py").read_text()
    forbidden = [
        "class LiteDataProxy",
        "class PositionProxy",
        "class IndicatorProxy",
        "class IndicatorBundle",
        "class _LineTA",
        "def _wrap_indicator_result",
    ]

    assert [token for token in forbidden if token in text] == []


def test_lite_modules_do_not_keep_backtesting_py_names() -> None:
    root = Path(__file__).parents[3]
    checked = [
        root / "tradelearn" / "lite" / "data.py",
        root / "tradelearn" / "lite" / "strategy.py",
    ]

    offenders = [
        str(path.relative_to(root))
        for path in checked
        if "Backtesting" in path.read_text()
    ]

    assert offenders == []


def test_lite_backtest_uses_public_broker_storage_api() -> None:
    root = Path(__file__).parents[3]
    text = (root / "tradelearn" / "lite" / "backtest.py").read_text()

    assert "broker._storage" not in text
