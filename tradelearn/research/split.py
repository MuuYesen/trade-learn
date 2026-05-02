"""Dataset split helpers for research workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradelearn.research.run import tracked


@tracked("split")
def time_split(
    data: pd.DataFrame | pd.Series,
    *,
    split: Any,
    level: str | int | None = None,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """Split time-indexed data into train/test sets.

    The split point belongs to the test set: ``train < split`` and
    ``test >= split``. MultiIndex panels use the first datetime-like level
    unless ``level`` is provided.
    """

    values = _datetime_values(data.index, level=level)
    split_ts = _align_timestamp(pd.Timestamp(split), values)
    train = data.loc[values < split_ts].copy()
    test = data.loc[values >= split_ts].copy()
    return train, test


def split_bars(
    bars: pd.DataFrame | pd.Series,
    *,
    split: Any,
    level: str | int | None = None,
) -> pd.DataFrame | pd.Series:
    """Return bar data on or after ``split`` for test-period backtests.

    This keeps research workflows explicit: full bars can be used to build
    factors and fit preprocessing, while only the test period enters backtest
    performance statistics.
    """

    values = _datetime_values(bars.index, level=level)
    split_ts = _align_timestamp(pd.Timestamp(split), values)
    return bars.loc[values >= split_ts].copy()


def _datetime_values(index: pd.Index, *, level: str | int | None) -> pd.Index:
    if isinstance(index, pd.MultiIndex):
        selected_level = _datetime_level(index, level=level)
        return pd.DatetimeIndex(index.get_level_values(selected_level))
    if _is_datetime_index(index):
        return pd.DatetimeIndex(index)
    raise ValueError("time_split requires a datetime index or datetime MultiIndex level")


def _datetime_level(index: pd.MultiIndex, *, level: str | int | None) -> str | int:
    if level is not None:
        values = index.get_level_values(level)
        if not _is_datetime_index(values):
            raise ValueError(f"MultiIndex level {level!r} is not datetime-like")
        return level

    preferred_names = {"timestamp", "datetime", "date", "time"}
    for name in index.names:
        if name in preferred_names and _is_datetime_index(index.get_level_values(name)):
            return name
    for i in range(index.nlevels):
        if _is_datetime_index(index.get_level_values(i)):
            return i
    raise ValueError("time_split requires a datetime MultiIndex level")


def _is_datetime_index(values: Any) -> bool:
    try:
        pd.DatetimeIndex(values)
    except (TypeError, ValueError):
        return False
    return True


def _align_timestamp(split: pd.Timestamp, values: pd.Index) -> pd.Timestamp:
    tz = getattr(values.dtype, "tz", None)
    if tz is None:
        return split.tz_localize(None) if split.tzinfo is not None else split
    if split.tzinfo is None:
        return split.tz_localize(tz)
    return split.tz_convert(tz)


__all__ = ["split_bars", "time_split"]
