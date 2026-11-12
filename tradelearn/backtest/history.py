"""Shared point-in-time snapshots for research and Lite adapters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import pandas as pd

DEFAULT_HISTORY_FIELDS = ("open", "high", "low", "close", "volume")


def window(
    source: Any, *, size: int, data: Any = None, ago: int = 0, fields: Sequence[str] | None = None
) -> pd.DataFrame:
    """Extract delivered bars; a feed returns time rows, a collection a panel.

    A strategy source supplies the common decision clock. Direct collections
    use their first feed's delivered clock. No alignment or filling is applied.
    """
    _bounds(size, ago)
    chosen = data if data is not None else getattr(source, "data", source)
    single = hasattr(chosen, "_frame") or hasattr(chosen, "_feed")
    feeds = [chosen] if single else list(chosen)
    feeds = [getattr(feed, "_feed", feed) for feed in feeds]
    clock = getattr(source, "_decision_datetime", None)
    if not callable(clock):
        clock = getattr(source, "current_datetime", None)
    cutoff = (
        pd.Timestamp(clock())
        if callable(clock)
        else _delivered_time(
            getattr(getattr(source, "data", None), "_feed", getattr(source, "data", None))
            or (feeds[0] if feeds else None)
        )
    )
    selected = DEFAULT_HISTORY_FIELDS if fields is None else tuple(fields)
    parts = [_extract(feed, size, ago, selected, cutoff) for feed in feeds]
    if single:
        return parts[0]
    return _panel(parts, selected)


def build_history_panel(
    datas: Iterable[Any],
    lookback: int | None = None,
    *,
    fields: Sequence[str] = DEFAULT_HISTORY_FIELDS,
    size: int | None = None,
    ago: int = 0,
) -> pd.DataFrame:
    """Compatibility panel entry; new callers should use bounded ``window``."""
    if size is not None and lookback is not None:
        raise ValueError("specify size or lookback, not both")
    size = lookback if size is None else size
    _bounds(size, ago, unbounded=True)
    feeds = [getattr(feed, "_feed", feed) for feed in datas]
    cutoff = _delivered_time(feeds[0]) if feeds else None
    parts = [
        _extract(feed, size, ago, fields, cutoff, fallback=f"data{i}")
        for i, feed in enumerate(feeds)
    ]
    return _panel(parts, fields)


def _bounds(size: int | None, ago: int, *, unbounded: bool = False) -> None:
    if not (unbounded and size is None) and (
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
    ):
        raise ValueError("size must be a positive integer")
    if isinstance(ago, bool) or not isinstance(ago, Integral) or ago > 0:
        raise ValueError("ago must be an integer <= 0")


def _delivered_time(feed: Any) -> pd.Timestamp | None:
    cursor = int(getattr(feed, "_cursor", -1))
    frame = getattr(feed, "_frame", None)
    if frame is None or cursor < 0:
        return None
    return pd.Timestamp(frame.index[min(cursor, len(frame) - 1)]) if len(frame) else None


def _extract(
    feed: Any,
    size: int | None,
    ago: int,
    fields: Sequence[str],
    cutoff: pd.Timestamp | None,
    fallback: str | None = None,
) -> pd.DataFrame:
    frame = getattr(feed, "_frame", None)
    if frame is None:
        raise TypeError("window requires a runtime data feed")
    symbol = getattr(feed, "_name", None) or frame.attrs.get("symbol") or fallback
    if not symbol:
        raise ValueError("window requires explicit feed symbol identity")
    missing = [field for field in fields if field not in frame]
    if missing:
        raise ValueError(f"missing window fields: {missing}")
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("feed requires sorted unique DatetimeIndex")
    end = min(len(frame), max(0, int(getattr(feed, "_cursor", -1)) + 1))
    if cutoff is None:
        end = 0
    else:
        if (frame.index.tz is None) != (cutoff.tzinfo is None):
            raise ValueError("feed and decision clock timezone mismatch")
        end = min(end, int(frame.index.searchsorted(cutoff, side="right")))
    end = max(0, end + ago)
    start = 0 if size is None else max(0, end - size)
    result = frame.iloc[start:end].loc[:, list(fields)].copy(deep=True)
    result.index.name = "timestamp"
    result.attrs = deepcopy(frame.attrs)
    result.attrs["symbol"] = str(symbol)
    return result


def _panel(parts: list[pd.DataFrame], fields: Sequence[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            columns=list(fields),
            index=pd.MultiIndex.from_arrays(
                [pd.DatetimeIndex([], name="timestamp"), pd.Index([], name="symbol")],
                names=["timestamp", "symbol"],
            ),
        )
    rows = []
    for part in parts:
        item = part.copy()
        item["symbol"] = part.attrs["symbol"]
        item = item.set_index("symbol", append=True)
        item.attrs = {}
        rows.append(item)
    result = pd.concat(rows).sort_index()
    if result.index.has_duplicates:
        raise ValueError("duplicate timestamp/symbol in window")
    return result
