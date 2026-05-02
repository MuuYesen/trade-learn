from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

DEFAULT_HISTORY_FIELDS = ("open", "high", "low", "close", "volume")


def build_history_panel(
    datas: Iterable[Any],
    lookback: int | None = None,
    *,
    fields: Sequence[str] = DEFAULT_HISTORY_FIELDS,
) -> pd.DataFrame:
    """Build a recent multi-data OHLCV panel from event-driven data feeds."""

    rows: list[pd.DataFrame] = []
    for index, data in enumerate(datas):
        cursor = int(getattr(data, "_cursor", -1))
        if cursor < 0:
            continue
        frame = getattr(data, "_frame", None)
        if frame is None:
            continue

        end = cursor + 1
        start = 0 if lookback is None else max(0, end - int(lookback))
        window = pd.DataFrame(frame).iloc[start:end]
        if window.empty:
            continue

        columns = [column for column in fields if column in window.columns]
        window = window.loc[:, columns].copy()
        window["timestamp"] = window.index
        window["symbol"] = str(getattr(data, "_name", None) or f"data{index}")
        rows.append(window.reset_index(drop=True))

    if not rows:
        return _empty_history_panel(fields)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.set_index(["timestamp", "symbol"]).sort_index()
    panel.index.names = ["timestamp", "symbol"]
    return panel


def _empty_history_panel(fields: Sequence[str]) -> pd.DataFrame:
    columns = ["timestamp", "symbol", *fields]
    return pd.DataFrame(columns=columns).set_index(["timestamp", "symbol"])
