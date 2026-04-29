from __future__ import annotations

from pathlib import Path

import pandas as pd

import tradelearn as tl
from tradelearn.lite import Backtest, Signal, SignalStrategy, Strategy


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


def test_lite_uses_backtrader_bar_indexing_with_1x_position_call() -> None:
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

    assert stats["# Trades"] == 1
    assert seen == {
        "close_now": 12.0,
        "close_prev": 11.0,
        "line_now": 3.0,
        "line_prev": 2.0,
        "sma_now": 11.5,
    }


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

    records = stats["_records"]
    assert records["signal"].dropna().tolist() == [12.0, 13.0, 14.0]


def test_lite_rejects_sl_tp_until_bracket_orders_are_implemented() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(1)

        def next(self) -> None:
            if len(self.data) == 2:
                self.buy(size=1, sl=9.0, tp=13.0)

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()
    orders = stats["_strategy"].orders

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

    assert len(stats["_strategy"].orders) == 3


def test_lite_signal_sugar_trades_on_positive_signal() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.sig = self.I(pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=self.data.index))
            self.signal(self.sig, kind="long")
            self.start_on_bar(1)

    stats = Backtest(_data(), LiteStrategy, cash=1000.0).run()

    assert stats["_strategy"].position().size > 0
    assert len(stats["_strategy"].orders) >= 1


def test_lite_exports_signal_strategy_names() -> None:
    assert SignalStrategy is Strategy
    wrapped = Signal([0.0, 1.0])
    assert wrapped[1] == 1.0


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
