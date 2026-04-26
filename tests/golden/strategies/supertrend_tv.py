"""Golden strategy adapter: TradingView Supertrend."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "supertrend_tv"


class SupertrendTvStrategy(GoldenAdapterBase):
    """Use close versus recent range midpoint as a compact Supertrend proxy."""

    def should_enter(self) -> bool:
        return float(self.data.close[0]) > self._range_midpoint(3)

    def should_exit(self) -> bool:
        return float(self.data.close[0]) < self._range_midpoint(3)
