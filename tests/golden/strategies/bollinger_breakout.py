"""Golden strategy adapter: Bollinger breakout."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "bollinger_breakout"


class BollingerBreakoutStrategy(GoldenAdapterBase):
    """Buy range breakout, close when price returns below range midpoint."""

    def should_enter(self) -> bool:
        highs = self._values(self.data.high, 3)
        return float(self.data.close[0]) >= max(highs)

    def should_exit(self) -> bool:
        return float(self.data.close[0]) < self._range_midpoint(3)
