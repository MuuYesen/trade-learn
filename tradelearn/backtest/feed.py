from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.backtest.data import DataContainer
from tradelearn.backtest.lines import LineSeries

_PANEL_SYMBOL_LEVELS = {"symbol", "ticker"}


def panel_symbol_level(data: Any) -> str | None:
    index = getattr(data, "index", None)
    names = getattr(index, "names", None)
    if not names or len(names) < 2:
        return None
    for level in names:
        if level and str(level).lower() in _PANEL_SYMBOL_LEVELS:
            return str(level)
    return None


def is_panel_ohlcv_frame(data: Any) -> bool:
    return (
        hasattr(data, "columns")
        and hasattr(data, "index")
        and panel_symbol_level(data) is not None
    )


def split_panel_ohlcv_frame(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    symbol_level = panel_symbol_level(data)
    if symbol_level is None:
        raise ValueError("panel data must have a symbol or ticker index level")
    return {
        str(symbol): data.xs(symbol, level=symbol_level, drop_level=True)
        for symbol in data.index.get_level_values(symbol_level).unique()
    }


def normalize_ohlcv_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize user OHLCV data into the shared backtest runtime shape."""
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if len(frame.columns.levels[0]) != 1:
            raise NotImplementedError(
                "multi-ticker MultiIndex frames should be passed as a dict of DataFrames"
            )
        frame = frame.xs(frame.columns.levels[0][0], axis=1, level=0)
    lower = {column: str(column).lower() for column in frame.columns}
    frame = frame.rename(columns=lower)
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"data must contain OHLC columns; missing {sorted(missing)}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if not isinstance(frame.index, pd.DatetimeIndex):
        numeric_index = pd.to_numeric(frame.index, errors="coerce")
        if not pd.isna(numeric_index).any() and numeric_index.max() > 10_000_000:
            frame.index = pd.to_datetime(numeric_index, unit="s", utc=True)
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame


def is_normalized_ohlcv_frame(data: pd.DataFrame) -> bool:
    """Return whether an OHLCV frame can skip normalization work."""
    if is_panel_ohlcv_frame(data):
        return False
    if isinstance(data.columns, pd.MultiIndex):
        return False
    columns = {str(column) for column in data.columns}
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(columns):
        return False
    if any(str(column) != str(column).lower() for column in data.columns):
        return False
    if not isinstance(data.index, pd.DatetimeIndex):
        return False
    return bool(data.index.is_monotonic_increasing)


class RuntimeDataFeed(DataContainer):
    """Shared runtime data feed with OHLCV LineSeries views."""

    lines = ("datetime", "open", "high", "low", "close", "volume")

    def __init__(
        self,
        data: pd.DataFrame,
        name: str | None = None,
        *,
        assume_normalized: bool = False,
        copy: bool = True,
        datetime_array: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> None:
        frame = data if assume_normalized else normalize_ohlcv_frame(data)
        DataContainer.__init__(
            self,
            frame,
            name=name,
            copy=copy,
            datetime_array=datetime_array,
        )
        buffer = self.shared_bar_buffer()

        self.datetime = LineSeries(
            self._datetime,
            is_datetime=True,
            buffer=buffer,
            buffer_name="datetime",
        )
        self.open = LineSeries(self._open, buffer=buffer, buffer_name="open")
        self.high = LineSeries(self._high, buffer=buffer, buffer_name="high")
        self.low = LineSeries(self._low, buffer=buffer, buffer_name="low")
        self.close = LineSeries(self._close, buffer=buffer, buffer_name="close")
        self.volume = LineSeries(self._volume, buffer=buffer, buffer_name="volume")
        self._lines_list = [
            self.datetime,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
        ]

    def _advance(self, cursor: int) -> None:
        self._cursor = cursor
        for line in self._lines_list:
            line._cursor = cursor


def build_data_feeds(
    data: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    feed_cls: type[RuntimeDataFeed] = RuntimeDataFeed,
    default_name: str = "Asset",
    assume_normalized: bool = False,
    copy: bool = True,
) -> list[RuntimeDataFeed]:
    """Build runtime feeds from a single DataFrame or a ticker->DataFrame mapping."""
    if is_panel_ohlcv_frame(data):
        data = split_panel_ohlcv_frame(data)
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data dict must contain at least one ticker")
        shared_datetime = _shared_datetime_array(data.values()) if assume_normalized else None
        return [
            feed_cls(
                frame,
                name=str(name),
                assume_normalized=assume_normalized,
                copy=copy,
                datetime_array=shared_datetime,
            )
            for name, frame in data.items()
        ]
    return [
        feed_cls(
            data,
            name=default_name,
            assume_normalized=assume_normalized,
            copy=copy,
        )
    ]


def _shared_datetime_array(frames: Any) -> np.ndarray | None:
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        return None
    if not isinstance(first.index, pd.DatetimeIndex):
        return None
    first_index = first.index
    for frame in iterator:
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(first_index):
            return None
    return first_index.values.astype("datetime64[s]").view(np.int64)
