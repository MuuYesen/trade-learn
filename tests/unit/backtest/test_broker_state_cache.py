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

    def observe_valuation(self, symbols, include_cash, settle):
        if symbols is not None:
            assert symbols == ['A'] and not include_cash and settle
            return (90.0, 20.0, 20.0)
        return (90.0, 100.0, 40.0)

    def drain_cash_events(self):
        return []

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


def test_broker_drain_rust_controls_fast_path_idle_and_populated() -> None:
    broker = RustBroker()
    # Idle state should short-circuit to None with no allocations.
    assert broker.drain_rust_controls() is None

    broker._queue_cash_control("add", 500.0)
    controls = broker.drain_rust_controls()
    assert controls is not None
    cash, checks, cancels, updates, expiry, market = controls
    assert len(cash) == 1
    assert checks == []
    assert cancels == []
    assert updates == []
    assert expiry == []
    assert market == []

    # Subsequent call when idle returns None again.
    assert broker.drain_rust_controls() is None


def test_rust_backtest_engine_accepts_contiguous_and_strided_numpy_arrays() -> None:
    import numpy as np
    from tradelearn._rust import RustBacktestEngine

    ts = np.array([1_700_000_000_000_000_000, 1_700_086_400_000_000_000], dtype=np.int64)
    prices = np.array([10.0, 11.0], dtype=np.float64)
    vol = np.array([1000.0, 1200.0], dtype=np.float64)
    kwargs = (100_000.0, 0.0, False, False, False, 0.0, 0.0, True, True, False)

    eng_contig = RustBacktestEngine(ts, prices, prices, prices, prices, vol, *kwargs)
    assert eng_contig.get_cash() == 100_000.0

    # Non-contiguous strided views should seamlessly fall back without error.
    ts_strided = np.repeat(ts, 2)[::2]
    prices_strided = np.repeat(prices, 2)[::2]
    vol_strided = np.repeat(vol, 2)[::2]
    assert not ts_strided.flags["C_CONTIGUOUS"]
    eng_strided = RustBacktestEngine(
        ts_strided,
        prices_strided,
        prices_strided,
        prices_strided,
        prices_strided,
        vol_strided,
        *kwargs,
    )
    assert eng_strided.get_cash() == 100_000.0
