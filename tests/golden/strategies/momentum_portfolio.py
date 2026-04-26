"""Golden strategy adapter: portfolio momentum."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "momentum_portfolio"


class MomentumPortfolioStrategy(GoldenAdapterBase):
    """Single-feed momentum proxy for portfolio golden expected runs."""

    def should_enter(self) -> bool:
        return self._momentum(2) > 0

    def should_exit(self) -> bool:
        return self._momentum(2) < 0
