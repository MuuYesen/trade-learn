"""Cross-sectional engine strategy helpers for index-enhancement workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradelearn.backtest.targets import TargetWeightSnapshot, build_target_weight_intents
from tradelearn.engine.strategy import Strategy


class IndexEnhanceStrategy(Strategy):
    """Engine cross-sectional strategy helper.

    Subclasses keep the normal ``next`` lifecycle. This class only provides
    utilities for building a current cross section, checking rebalance
    windows, and translating symbol weights into order_target_percent() calls.
    """

    rebalance_freq: str | int = "monthly"
    close_missing: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_rebalance_key: Any = None

    def current_universe(self) -> pd.DataFrame:
        """Return the current multi-data OHLCV cross section."""

        rows: list[dict[str, Any]] = []
        for data in self.datas:
            symbol = str(getattr(data, "_name", None) or f"data{len(rows)}")
            rows.append(
                {
                    "symbol": symbol,
                    "datetime": _data_datetime(data),
                    "open": float(data.open[0]),
                    "high": float(data.high[0]),
                    "low": float(data.low[0]),
                    "close": float(data.close[0]),
                    "volume": float(data.volume[0]),
                }
            )
        frame = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
        if not frame.empty:
            frame.index.name = "symbol"
        return frame

    def target_weights(
        self,
        weights: Mapping[str, float] | pd.Series,
        *,
        close_missing: bool = True,
    ) -> list[Any]:
        """Move all data feeds toward symbol target weights."""

        data_by_name = self._data_by_name()
        snapshots = {
            name: TargetWeightSnapshot(
                price=float(data.close[0]),
                size=float(self.getposition(data).size + self._pending_size.get(data, 0.0)),
                mult=self._position_mult(data),
            )
            for name, data in data_by_name.items()
        }
        intents = build_target_weight_intents(
            weights,
            data_by_symbol=data_by_name,
            snapshots=snapshots,
            equity=float(self.broker.getvalue()) if self.broker is not None else 0.0,
            close_missing=close_missing,
            unknown_label="symbol(s)",
        )
        orders: list[Any] = []
        for intent in intents:
            order = self.order_target_percent(data=intent.data, target=intent.target_weight)
            if order is not None:
                orders.append(order)
        return orders

    def current_datetime(self) -> pd.Timestamp:
        """Return the timestamp of the current primary data bar."""

        return self._current_datetime()

    def should_rebalance(self) -> bool:
        """Return whether the current bar starts a new rebalance window."""

        return self._should_rebalance()

    def _data_by_name(self) -> dict[str, Any]:
        return {
            str(getattr(data, "_name", None) or f"data{i}"): data
            for i, data in enumerate(self.datas)
        }

    def _current_datetime(self) -> pd.Timestamp:
        if self.data is None:
            return pd.NaT
        return pd.Timestamp(_data_datetime(self.data))

    def _should_rebalance(self) -> bool:
        cursor = int(getattr(self.data, "_cursor", len(self.data) - 1))
        freq = self.rebalance_freq
        if isinstance(freq, int):
            if freq <= 0:
                raise ValueError("integer rebalance_freq must be positive")
            return cursor % freq == 0

        key = self._rebalance_key(str(freq).lower())
        if key == self._last_rebalance_key:
            return False
        self._last_rebalance_key = key
        return True

    def _rebalance_key(self, freq: str) -> Any:
        dt = self._current_datetime()
        if freq in {"daily", "day", "d"}:
            return dt.date()
        if freq in {"weekly", "week", "w"}:
            iso = dt.isocalendar()
            return iso.year, iso.week
        if freq in {"monthly", "month", "m"}:
            return dt.year, dt.month
        if freq in {"quarterly", "quarter", "q"}:
            return dt.year, (dt.month - 1) // 3 + 1
        if freq in {"yearly", "annual", "year", "y"}:
            return dt.year
        raise ValueError(f"unsupported rebalance_freq: {self.rebalance_freq!r}")


def _data_datetime(data: Any) -> pd.Timestamp:
    line = getattr(data, "datetime", None)
    if line is not None and hasattr(line, "datetime"):
        return pd.Timestamp(line.datetime(0))
    frame = getattr(data, "_frame", None)
    cursor = int(getattr(data, "_cursor", 0))
    if frame is not None and len(frame.index):
        return pd.Timestamp(frame.index[cursor])
    return pd.NaT
