from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradelearn.lite.indicator import IndicatorProxy


class LiteDataProxy:
    """Tradelearn Lite data view with lower-case OHLCV lines."""

    __slots__ = (
        "_feed",
        "_open_array",
        "_high_array",
        "_low_array",
        "_close_array",
        "_volume_array",
        "_open_proxy",
        "_high_proxy",
        "_low_proxy",
        "_close_proxy",
        "_volume_proxy",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "_extra_line_cache",
    )

    def __init__(self, data_feed: Any):
        self._feed = data_feed
        self._open_array = data_feed.get_array("open")
        self._high_array = data_feed.get_array("high")
        self._low_array = data_feed.get_array("low")
        self._close_array = data_feed.get_array("close")
        self._volume_array = data_feed.get_array("volume")
        frame = _ta_frame(data_feed)
        self._open_proxy = IndicatorProxy(
            self._open_array, data_feed, index=frame.index, name="open"
        )
        self._high_proxy = IndicatorProxy(
            self._high_array, data_feed, index=frame.index, name="high"
        )
        self._low_proxy = IndicatorProxy(self._low_array, data_feed, index=frame.index, name="low")
        self._close_proxy = IndicatorProxy(
            self._close_array, data_feed, index=frame.index, name="close"
        )
        self._volume_proxy = IndicatorProxy(
            self._volume_array, data_feed, index=frame.index, name="volume"
        )
        self.open = self._open_proxy
        self.high = self._high_proxy
        self.low = self._low_proxy
        self.close = self._close_proxy
        self.volume = self._volume_proxy
        self._extra_line_cache: dict[str, tuple[Any, IndicatorProxy]] = {}

    def __getattr__(self, name: str) -> Any:
        if name[:1].isupper():
            raise AttributeError(
                f"Column '{name}' is not available in the Tradelearn Lite facade; "
                f"use '{name.lower()}' instead."
            )
        line = self._line_or_array(name)
        if len(line._data) == 0:
            raise AttributeError(f"Column '{name}' not in data")
        return line

    def _line_or_array(self, core_name: str) -> Any:
        arr = self._feed.get_array(core_name)
        cached = self._extra_line_cache.get(core_name)
        if cached is not None and cached[0] is arr:
            return cached[1]
        line = IndicatorProxy(arr, self._feed, index=_ta_frame(self._feed).index, name=core_name)
        self._extra_line_cache[core_name] = (arr, line)
        return line

    def __len__(self) -> int:
        return self._feed._cursor + 1

    @property
    def df(self) -> pd.DataFrame:
        frame = _ta_frame(self._feed)
        cursor = self._feed._cursor
        if cursor < 0:
            return frame
        return frame.iloc[: cursor + 1]

    @property
    def index(self) -> pd.Index:
        return self.df.index

    @property
    def now(self) -> Any:
        return self.index[-1]

    @property
    def tickers(self) -> list[str]:
        frame = _ta_frame(self._feed)
        if isinstance(frame.columns, pd.MultiIndex):
            return [str(value) for value in frame.columns.get_level_values(0).unique()]
        return [str(getattr(self._feed, "_name", None) or "Asset")]

    @property
    def the_ticker(self) -> str:
        tickers = self.tickers
        if len(tickers) != 1:
            raise ValueError("Ticker must explicitly specified for multi-asset Lite backtests")
        return tickers[0]

    @property
    def pip(self) -> float:
        close = np.asarray(self._close_array, dtype=float)
        if close.size == 0:
            return 0.01
        decimals = [
            len(str(value).partition(".")[-1].rstrip("0"))
            for value in close
            if np.isfinite(value)
        ]
        if not decimals:
            return 0.01
        return float(10 ** -int(np.median(decimals)))


def _ta_frame(data_feed: Any) -> pd.DataFrame:
    frame = getattr(data_feed, "_frame", None)
    if frame is not None:
        return frame
    return pd.DataFrame(
        {
            "open": data_feed.get_array("open"),
            "high": data_feed.get_array("high"),
            "low": data_feed.get_array("low"),
            "close": data_feed.get_array("close"),
            "volume": data_feed.get_array("volume"),
        }
    )


__all__ = ["LiteDataProxy", "_ta_frame"]
