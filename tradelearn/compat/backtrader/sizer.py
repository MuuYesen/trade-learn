from __future__ import annotations
from typing import Any
from tradelearn.backtest.sizer import Sizer as CoreSizer
from tradelearn.backtest.models import Params

class Sizer(CoreSizer):
    """Backtrader-style Sizer with params support."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cls_params = getattr(self.__class__, 'params', [])
        self.params = self.p = Params(cls_params, **kwargs)

class FixedSize(Sizer):
    params = (("stake", 1),)
    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        return float(self.p.stake)

class PercentSizer(Sizer):
    params = (("percents", 10.0),)
    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        value = self.broker.getvalue()
        price = data.close[0]
        if price <= 0: return 0.0
        return (value * self.p.percents / 100.0) / price

class AllInSizer(Sizer):
    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        cash = self.broker.getcash()
        price = data.close[0]
        if price <= 0: return 0.0
        return cash / price
