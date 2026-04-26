"""Golden strategy adapter: ML plus Alpha101."""

from tests.golden.strategies._helpers import GoldenAdapterBase

STRATEGY_NAME = "alpha101_ml"


class Alpha101MlStrategy(GoldenAdapterBase):
    """Use a compact alpha proxy based on recent momentum and range position."""

    def should_enter(self) -> bool:
        return self._momentum() > 0 and float(self.data.close[0]) > self._range_midpoint(3)

    def should_exit(self) -> bool:
        return self._momentum() < 0
