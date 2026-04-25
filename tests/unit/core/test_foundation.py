from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd
import pytest

from tradelearn.core import (
    ContractError,
    Experiment,
    StreamBar,
    TradelearnError,
    ensure_utc,
    get_seed,
    progress,
    set_global_seed,
    utc_now,
    validate_bars,
    validate_returns,
)


def test_error_hierarchy() -> None:
    assert issubclass(ContractError, TradelearnError)


def test_ensure_utc_converts_naive_and_aware_timestamps() -> None:
    naive = ensure_utc("2026-04-25 12:00:00")
    aware = ensure_utc(pd.Timestamp("2026-04-25 21:00:00", tz="Asia/Tokyo"))

    assert str(naive.tz) == "UTC"
    assert str(aware.tz) == "UTC"
    assert aware.hour == 12
    assert utc_now().tzinfo is not None


def test_seed_reads_and_sets_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADELEARN_SEED", raising=False)
    assert get_seed() is None

    set_global_seed(7)
    assert os.environ["TRADELEARN_SEED"] == "7"
    assert get_seed() == 7


def test_progress_returns_iterable() -> None:
    values = progress([1, 2, 3], desc="unit")
    assert isinstance(values, Iterable)
    assert list(values) == [1, 2, 3]


def test_contract_dataclasses() -> None:
    bar = StreamBar(
        ts=ensure_utc("2026-04-25"),
        symbol="GOOG",
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=100.0,
    )
    exp = Experiment(name="exp", params={"fast": 10}, metrics={"sharpe": 1.0})

    assert bar.symbol == "GOOG"
    assert exp.params["fast"] == 10


def test_validate_bars_accepts_contract_shape() -> None:
    index = pd.MultiIndex.from_tuples(
        [(ensure_utc("2026-04-25"), "GOOG")],
        names=["timestamp", "symbol"],
    )
    bars = pd.DataFrame(
        {
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [11.0],
            "volume": [1000.0],
        },
        index=index,
    )

    assert validate_bars(bars) is bars


def test_validate_bars_rejects_bad_ohlc() -> None:
    index = pd.MultiIndex.from_tuples(
        [(ensure_utc("2026-04-25"), "GOOG")],
        names=["timestamp", "symbol"],
    )
    bars = pd.DataFrame(
        {
            "open": [10.0],
            "high": [9.0],
            "low": [8.0],
            "close": [11.0],
            "volume": [1000.0],
        },
        index=index,
    )

    with pytest.raises(ContractError, match="OHLC"):
        validate_bars(bars)


def test_validate_returns_requires_utc_datetime_index() -> None:
    returns = pd.Series(
        [0.01, -0.02],
        index=pd.DatetimeIndex(
            [ensure_utc("2026-04-24"), ensure_utc("2026-04-25")],
            name="timestamp",
        ),
    )
    assert validate_returns(returns) is returns

    bad = pd.Series([0.01], index=pd.DatetimeIndex(["2026-04-25"]))
    with pytest.raises(ContractError, match="tz-aware UTC"):
        validate_returns(bad)
