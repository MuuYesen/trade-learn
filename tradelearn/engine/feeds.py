"""Backtrader-style data feed adapters."""

from __future__ import annotations

import pandas as pd

from .datafeed import DataFeed


class PandasData(DataFeed):
    """Map a pandas DataFrame to the OHLCV lines expected by Strategy."""

    def __init__(
        self,
        *,
        dataname: pd.DataFrame,
        name: str | None = None,
        datetime: str | None = None,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ) -> None:
        frame = dataname.copy()
        if datetime is not None:
            frame = frame.set_index(datetime)
        mapped = pd.DataFrame(
            {
                "open": frame[open],
                "high": frame[high],
                "low": frame[low],
                "close": frame[close],
                "volume": frame[volume],
            },
            index=frame.index,
        )
        super().__init__(mapped, name=name)


__all__ = ["PandasData"]
