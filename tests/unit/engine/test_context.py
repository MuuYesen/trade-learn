from __future__ import annotations

import tradelearn as bt
from tradelearn.engine.base import _G, engine_context


def test_engine_context_restores_nested_state() -> None:
    outer_data = object()
    outer_datas = [outer_data]
    outer_strategy = object()
    inner_data = object()
    inner_strategy = object()

    with engine_context(data=outer_data, datas=outer_datas, strategy=outer_strategy):
        assert _G.current_data is outer_data
        assert _G.current_datas == outer_datas
        assert _G.current_strategy is outer_strategy

        with engine_context(data=inner_data, datas=[inner_data], strategy=inner_strategy):
            assert _G.current_data is inner_data
            assert _G.current_datas == [inner_data]
            assert _G.current_strategy is inner_strategy

        assert _G.current_data is outer_data
        assert _G.current_datas == outer_datas
        assert _G.current_strategy is outer_strategy

    assert _G.current_data is None
    assert _G.current_datas == []
    assert _G.current_strategy is None


def test_indicator_created_inside_strategy_init_uses_strategy_data_context() -> None:
    from tradelearn.engine import Indicator

    class Mid(Indicator):
        lines = ("mid",)

        def __init__(self) -> None:
            self.lines.mid = (self.data.high + self.data.low) / 2

    class UsesCustomIndicator(bt.Strategy):
        def __init__(self) -> None:
            self.mid = Mid()
            self.values: list[float] = []

        def next(self) -> None:
            self.values.append(self.mid.mid[0])

    import pandas as pd

    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 14.0],
            "low": [8.0, 9.0, 10.0],
            "close": [11.0, 12.0, 13.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(frame)
    cerebro.addstrategy(UsesCustomIndicator)

    [strategy] = cerebro.run()

    assert strategy.values == [10.0, 11.0, 12.0]
