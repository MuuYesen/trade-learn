"""Golden strategy adapter: equal-weight rotation."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "equal_weight"


class EqualWeightStrategy(GoldenAdapterBase):
    """Hold a fixed unit after warmup for single-feed TV subset runs."""

    def should_enter(self) -> bool:
        return True

    def should_exit(self) -> bool:
        return False
