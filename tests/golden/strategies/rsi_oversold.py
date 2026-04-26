"""Golden strategy adapter: RSI oversold reversal."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "rsi_oversold"


class RsiOversoldStrategy(GoldenAdapterBase):
    """Buy after a short negative swing, close after rebound."""

    def should_enter(self) -> bool:
        return self._momentum() < 0

    def should_exit(self) -> bool:
        return self._momentum() > 0
