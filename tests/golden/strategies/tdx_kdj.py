"""Golden strategy adapter: A-share KDJ."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "tdx_kdj"


class TdxKdjStrategy(GoldenAdapterBase):
    """Use close location in recent high/low range as a KDJ proxy."""

    def _k_value(self) -> float:
        highs = self._values(self.data.high, 3)
        lows = self._values(self.data.low, 3)
        span = max(highs) - min(lows)
        if span == 0:
            return 50.0
        return (float(self.data.close[0]) - min(lows)) / span * 100.0

    def should_enter(self) -> bool:
        return self._k_value() > 50.0

    def should_exit(self) -> bool:
        return self._k_value() < 50.0
