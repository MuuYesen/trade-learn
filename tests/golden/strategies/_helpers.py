"""Shared deterministic adapters for TV subset golden expected generation."""

from __future__ import annotations

from statistics import fmean

from tradelearn.compat.backtrader import Strategy


class GoldenAdapterBase(Strategy):
    """Small single-asset adapter base for Stage 0 golden generation."""

    min_period = 3
    size = 1.0

    def _values(self, line: object, size: int) -> list[float]:
        return [float(value) for value in line.get(size=size)]

    def _sma(self, size: int) -> float:
        values = self._values(self.data.close, size)
        return fmean(values) if values else float(self.data.close[0])

    def _momentum(self, size: int = 2) -> float:
        values = self._values(self.data.close, size + 1)
        if len(values) <= size:
            return 0.0
        return values[-1] - values[0]

    def _range_midpoint(self, size: int = 3) -> float:
        highs = self._values(self.data.high, size)
        lows = self._values(self.data.low, size)
        return (max(highs) + min(lows)) / 2.0

    def should_enter(self) -> bool:
        return self._momentum() > 0

    def should_exit(self) -> bool:
        return self._momentum() < 0

    def next(self) -> None:
        if not self.position and self.should_enter():
            self.buy(size=self.size)
        elif self.position and self.should_exit():
            self.close()
