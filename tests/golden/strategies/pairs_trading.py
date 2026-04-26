"""Golden strategy adapter: multi-asset pairs trading."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "pairs_trading"


class PairsTradingStrategy(GoldenAdapterBase):
    """Single-feed proxy for spread mean reversion in TV subset expected runs."""

    def should_enter(self) -> bool:
        return float(self.data.close[0]) < self._sma(3)

    def should_exit(self) -> bool:
        return float(self.data.close[0]) > self._sma(3)
