from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradelearn.backtest.data import DataContainer
from tradelearn.backtest.lines import LineSeries


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


class RuntimeDataFeed(DataContainer):
    """Shared runtime data feed with OHLCV LineSeries views."""

    lines = ("datetime", "open", "high", "low", "close", "volume")

    def __init__(self, data: pd.DataFrame, name: str | None = None, **_kwargs: Any) -> None:
        DataContainer.__init__(self, normalize_ohlcv_frame(data), name=name)
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
) -> list[RuntimeDataFeed]:
    """Build runtime feeds from a single DataFrame or a ticker->DataFrame mapping."""
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data dict must contain at least one ticker")
        return [feed_cls(frame, name=str(name)) for name, frame in data.items()]
    return [feed_cls(data, name=default_name)]
