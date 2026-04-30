"""Cross-sectional engine strategy helpers for index-enhancement workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tradelearn.engine.strategy import Strategy


class IndexEnhanceStrategy(Strategy):
    """Backtrader-style cross-sectional strategy base.

    Subclasses override :meth:`rebalance` and return target weights by data
    feed name. The class keeps the normal event-driven ``next`` lifecycle:
    it only adds a thin rebalance trigger and translates weights into existing
    ``order_target_percent`` calls.
    """

    rebalance_freq: str | int = "monthly"
    close_missing: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_rebalance_key: Any = None

    def next(self) -> None:
        if not self._should_rebalance():
            return
        dt = self._current_datetime()
        weights = self.rebalance(dt, self.current_universe())
        self.target_weights(weights, close_missing=self.close_missing)

    def rebalance(
        self,
        dt: pd.Timestamp,
        universe: pd.DataFrame,
    ) -> Mapping[str, float] | pd.Series:
        """Return target weights for the current cross section.

        Parameters
        ----------
        dt
            Current bar timestamp.
        universe
            Symbol-indexed OHLCV snapshot built from all bound data feeds.
        """

        return {}

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

        requested = _coerce_weights(weights)
        cash_weight = requested.pop("cash", 0.0)
        if cash_weight < 0:
            raise ValueError("cash target weight must be non-negative")
        if any(weight < 0 for weight in requested.values()):
            raise ValueError("target weights must be non-negative")
        if sum(requested.values()) + cash_weight > 1.000000000000001:
            raise ValueError("target weights plus cash must sum to <= 1")

        data_by_name = self._data_by_name()
        unknown = sorted(set(requested) - set(data_by_name))
        if unknown:
            raise ValueError(f"Unknown symbol(s): {unknown}")

        targets = dict(requested)
        if close_missing:
            for name in data_by_name.keys() - targets.keys():
                targets[name] = 0.0

        ordered = sorted(
            targets.items(),
            key=lambda item: (item[1] > self._current_weight(data_by_name[item[0]]), item[0]),
        )
        orders: list[Any] = []
        for name, target in ordered:
            order = self.order_target_percent(data=data_by_name[name], target=float(target))
            if order is not None:
                orders.append(order)
        return orders

    def _data_by_name(self) -> dict[str, Any]:
        return {
            str(getattr(data, "_name", None) or f"data{i}"): data
            for i, data in enumerate(self.datas)
        }

    def _current_weight(self, data: Any) -> float:
        total = float(self.broker.getvalue()) if self.broker is not None else 0.0
        if total == 0:
            return 0.0
        return float(self.broker.getvalue(datas=[data])) / total

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


def _coerce_weights(weights: Mapping[str, float] | pd.Series) -> dict[str, float]:
    items = weights.items() if hasattr(weights, "items") else dict(weights).items()
    return {str(name): float(weight) for name, weight in items}


def _data_datetime(data: Any) -> pd.Timestamp:
    line = getattr(data, "datetime", None)
    if line is not None and hasattr(line, "datetime"):
        return pd.Timestamp(line.datetime(0))
    frame = getattr(data, "_frame", None)
    cursor = int(getattr(data, "_cursor", 0))
    if frame is not None and len(frame.index):
        return pd.Timestamp(frame.index[cursor])
    return pd.NaT
