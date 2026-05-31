"""Shared report and market replay glue for user facades."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tradelearn.report import Reporter


def market_data_from_datas(datas: Sequence[Any] | None) -> Any | None:
    """Return market data frames for report rendering."""

    if not datas:
        return None
    frames: dict[str, Any] = {}
    for index, data in enumerate(datas):
        frame = getattr(data, "_frame", None)
        if frame is None:
            continue
        name = str(getattr(data, "_name", None) or f"data{index}")
        frames[name] = frame
    if not frames:
        return None
    if len(frames) == 1:
        return next(iter(frames.values()))
    return frames


def reporter_from_stats(stats: Any, datas: Sequence[Any] | None) -> Reporter:
    """Build a Reporter from stats and facade data feeds."""

    return Reporter(stats, market_data=market_data_from_datas(datas))


def reporter_from_results(results: Sequence[Any] | None, datas: Sequence[Any] | None) -> Reporter:
    """Build a Reporter from the most recent run results."""

    if not results:
        raise RuntimeError("run() must be called before plot() or report()")
    stats = getattr(results[0], "stats", None)
    if stats is None:
        raise RuntimeError("last run did not produce stats")
    return reporter_from_stats(stats, datas)
