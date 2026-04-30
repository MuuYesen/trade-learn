from __future__ import annotations

import pandas as pd

from tradelearn import engine as bt


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [10.0, 11.0, 12.0, 13.0, 14.0],
            "low": [10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0] * 5,
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
    )


def _run_with_signal(sigtype: int, signal_cls: type, *, strategy: type | None = None, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.adddata(_data(), name="primary")
    if strategy is not None:
        cerebro.addstrategy(strategy)
    cerebro.add_signal(sigtype, signal_cls, **kwargs)
    return cerebro, cerebro.run()[0]


class ZeroSignal:
    def __getitem__(self, index: int) -> float:
        return 0.0


class LongSignal:
    def __getitem__(self, index: int) -> float:
        return 1.0


class ShortSignal:
    def __getitem__(self, index: int) -> float:
        return -1.0


class InvalidSignal:
    def __getitem__(self, index: int) -> str:
        return "invalid"


def test_signal_strategy_does_not_trade_on_zero_signal() -> None:
    cerebro, strategy = _run_with_signal(bt.SIGNAL_LONG, ZeroSignal)

    assert len(strategy.stats.fills) == 0
    assert cerebro.broker.getvalue() == 100000.0


def test_signal_strategy_trades_on_positive_long_signal() -> None:
    _cerebro, strategy = _run_with_signal(bt.SIGNAL_LONG, LongSignal)

    assert len(strategy.stats.fills) > 0
    assert strategy.position.size > 0


def test_signal_long_any_holds_while_signal_stays_nonzero() -> None:
    _cerebro, strategy = _run_with_signal(bt.SIGNAL_LONG_ANY, LongSignal)

    assert len(strategy.stats.fills) == 1
    assert strategy.position.size > 0


def test_signal_short_any_holds_while_signal_stays_nonzero() -> None:
    _cerebro, strategy = _run_with_signal(bt.SIGNAL_SHORT_ANY, ShortSignal)

    assert len(strategy.stats.fills) == 1
    assert strategy.position.size < 0


def test_signal_strategy_treats_invalid_signal_values_as_neutral() -> None:
    cerebro, strategy = _run_with_signal(bt.SIGNAL_LONG, InvalidSignal)

    assert len(strategy.stats.fills) == 0
    assert cerebro.broker.getvalue() == 100000.0


def test_add_signal_uses_signal_strategy_when_regular_strategy_exists() -> None:
    class EmptyStrategy(bt.Strategy):
        def next(self) -> None:
            pass

    _cerebro, strategy = _run_with_signal(bt.SIGNAL_LONG, LongSignal, strategy=EmptyStrategy)

    assert isinstance(strategy, bt.SignalStrategy)
    assert len(strategy.stats.fills) > 0


def test_signal_strategy_preserves_registered_args_and_kwargs() -> None:
    seen: dict[str, object] = {}

    class CustomSignalStrategy(bt.SignalStrategy):
        def __init__(self, label: str, *, threshold: int) -> None:
            seen["label"] = label
            seen["threshold"] = threshold
            super().__init__()

    cerebro = bt.Cerebro()
    cerebro.adddata(_data())
    cerebro.signal_strategy(CustomSignalStrategy, "custom", threshold=7)
    cerebro.add_signal(bt.SIGNAL_LONG, LongSignal)
    strategy = cerebro.run()[0]

    assert isinstance(strategy, CustomSignalStrategy)
    assert seen == {"label": "custom", "threshold": 7}


def test_signal_strategy_resolves_string_data_target() -> None:
    cerebro = bt.Cerebro()
    data = cerebro.adddata(_data(), name="primary")
    cerebro.signal_strategy(bt.SignalStrategy, _data="primary")
    cerebro.add_signal(bt.SIGNAL_LONG, LongSignal)
    strategy = cerebro.run()[0]

    assert strategy._dtarget is data
    assert len(strategy.stats.fills) > 0


def test_signal_strategy_subclass_next_runs_after_signal_processing() -> None:
    seen: list[int] = []

    class CustomSignalStrategy(bt.SignalStrategy):
        def next(self) -> None:
            seen.append(len(self.data))

    cerebro = bt.Cerebro()
    cerebro.adddata(_data())
    cerebro.signal_strategy(CustomSignalStrategy)
    cerebro.add_signal(bt.SIGNAL_LONG, LongSignal)
    strategy = cerebro.run()[0]

    assert seen
    assert len(strategy.stats.fills) > 0


def test_signal_strategy_auto_insertion_does_not_mutate_registered_strategies() -> None:
    cerebro = bt.Cerebro()
    cerebro.adddata(_data())
    cerebro.add_signal(bt.SIGNAL_LONG, LongSignal)

    cerebro.run()
    cerebro.run()

    assert cerebro.strats == []
