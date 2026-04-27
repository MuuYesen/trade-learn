from __future__ import annotations
from typing import Any
from tradelearn.backtest.base import BaseSizer

class Sizer(BaseSizer):
    """Base sizer class to determine order sizes automatically."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from tradelearn.backtest.base import Params
        cls_params = getattr(self.__class__, 'params', [])
        self.params = self.p = Params(cls_params, **kwargs)
        self.strategy: Any = None
        self.broker: Any = None

    def _set(self, strategy: Any, broker: Any) -> None:
        self.strategy = strategy
        self.broker = broker

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        return 0.0


class FixedSize(Sizer):
    """Sizer that returns a fixed size."""
    params = (("stake", 1),)

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        return float(self.p.stake)


class PercentSizer(Sizer):
    """Sizer that returns a percentage of total value."""
    params = (("percents", 10.0),)

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        value = self.broker.getvalue()
        price = data.close[0]
        if price <= 0:
            return 0.0
        return (value * self.p.percents / 100.0) / price


class AllInSizer(Sizer):
    """Sizer that uses all available cash."""

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        cash = self.broker.getcash()
        price = data.close[0]
        if price <= 0:
            return 0.0
        return cash / price
