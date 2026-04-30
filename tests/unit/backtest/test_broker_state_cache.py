from __future__ import annotations

from tradelearn.backtest.broker import RustBroker


class _CloseLine:
    def __getitem__(self, index: int) -> float:
        return 10.0


class _Data:
    close = _CloseLine()

    def __init__(self, name: str):
        self._name = name


class _Engine:
    def __init__(self):
        self.position_calls = 0

    def get_position_for_symbol(self, symbol: str) -> tuple[float, float]:
        self.position_calls += 1
        return (2.0, 5.0)

    def get_cash(self) -> float:
        return 90.0

    def get_position(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_equity(self) -> float:
        return 100.0


def test_broker_position_cache_reuses_symbol_lookup_within_bar() -> None:
    broker = RustBroker()
    engine = _Engine()
    first = _Data("A")
    second = _Data("B")
    broker.bind_engine(engine)
    broker.bind_datas([first, second])
    broker._curr_idx = 7

    assert broker.getposition(first).size == 2.0
    assert broker.getposition(first).price == 5.0
    assert broker.getvalue(datas=[first]) == 20.0
    assert engine.position_calls == 1

    broker._clear_state_caches()
    assert broker.getposition(first).size == 2.0
    assert engine.position_calls == 2
