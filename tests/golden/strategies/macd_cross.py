"""Golden strategy adapter: MACD crossover."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "macd_cross"


class MacdCrossStrategy(GoldenAdapterBase):
    """Use short-minus-long SMA as a compact MACD proxy."""

    def _macd_proxy(self) -> float:
        return self._sma(2) - self._sma(3)

    def should_enter(self) -> bool:
        return self._macd_proxy() > 0

    def should_exit(self) -> bool:
        return self._macd_proxy() < 0
