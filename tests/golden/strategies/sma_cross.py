"""Golden strategy adapter: moving average crossover."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "sma_cross"


class SmaCrossStrategy(GoldenAdapterBase):
    """Buy when a fast SMA is above a slow SMA, close on reversal."""

    min_period = 3

    def should_enter(self) -> bool:
        return self._sma(2) > self._sma(3)

    def should_exit(self) -> bool:
        return self._sma(2) < self._sma(3)
