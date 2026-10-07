"""Core contract objects and validators from ``design/specs/CONTRACTS.md``.

Only cross-runtime contracts belong here. Backtest-specific runtime models such
as ``Order``, ``Trade``, and ``ExecutedInfo`` stay in the backtest package until
they are split into broker-neutral contracts for backtest, paper, and live
trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.core.errors import ContractError


@dataclass(frozen=True)
class StreamBar:
    """Single immutable streaming bar."""

    ts: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


@dataclass
class Experiment:
    """Internal MLflow run abstraction used by analyzers."""

    name: str
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    run_id: str | None = None
    parent_run_id: str | None = None


REQUIRED_BAR_COLUMNS = ("open", "high", "low", "close", "volume")


def _require_utc_index(index: pd.DatetimeIndex, label: str) -> None:
    """Raise when an index is not UTC-aware."""

    if index.tz is None or str(index.tz) != "UTC":
        raise ContractError(f"{label} index must be tz-aware UTC")


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate the Stage 0-enforceable Bars contract."""

    if not isinstance(bars.index, pd.MultiIndex):
        raise ContractError("Bars index must be a MultiIndex(timestamp, symbol)")
    if bars.index.nlevels != 2:
        raise ContractError("Bars index must have two levels: timestamp and symbol")
    if bars.index.has_duplicates:
        raise ContractError("Bars index contains duplicate timestamp/symbol rows")
    missing = [col for col in REQUIRED_BAR_COLUMNS if col not in bars.columns]
    if missing:
        raise ContractError(f"Bars missing required columns: {missing}")

    timestamps = bars.index.get_level_values(0)
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    _require_utc_index(timestamps, "Bars timestamp")

    low_ok = bars["low"] <= np.minimum(bars["open"], bars["close"])
    high_ok = np.maximum(bars["open"], bars["close"]) <= bars["high"]
    if not bool((low_ok & high_ok).all()):
        raise ContractError("Bars OHLC invariant failed")
    return bars


def validate_returns(returns: pd.Series) -> pd.Series:
    """Validate the Stage 0-enforceable Returns contract."""

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ContractError("Returns index must be a DatetimeIndex")
    _require_utc_index(returns.index, "Returns")
    if not returns.index.is_monotonic_increasing:
        raise ContractError("Returns index must be sorted ascending")
    if bool(np.isinf(returns.to_numpy(dtype=float, na_value=np.nan)).any()):
        raise ContractError("Returns must not contain inf values")
    return returns
