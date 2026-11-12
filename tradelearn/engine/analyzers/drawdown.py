"""BT value-mode DrawDown, incrementally observing broker equity.

This is unadjusted observed-value drawdown, in BT's positive percent/money
units. It is not cashflow-adjusted investment performance and does not change
common Stats/Reporter ratio units or the public financial metrics contract.
"""
from __future__ import annotations

from tradelearn.metrics.risk import _drawdown_observation

from ..analyzer import Analyzer


class _Analysis(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class Drawdown(Analyzer):
    """Constant-state BT DrawDown; fund mode needs actual broker fund facts."""

    params = (("fund", None),)
    _metric_key = "drawdown"
    _is_streaming = True
    _requires_account_value = True

    def create_analysis(self):
        self.rets = _Analysis(len=0, drawdown=0.0, moneydown=0.0,
                              max=_Analysis(len=0.0, drawdown=0.0, moneydown=0.0))
        self._peak = None
        self._pending_value = None

    def start(self):
        self._fundmode = (getattr(self.strategy.broker, "fundmode", False)
                          if self.p.fund is None else self.p.fund)
        if type(self._fundmode) is not bool:
            raise ValueError("fund mode must be boolean")
        if self._fundmode and not hasattr(self.strategy.broker, "fundvalue"):
            raise NotImplementedError("fund-mode DrawDown requires authoritative broker fundvalue")

    def _observe_value(self, value):
        self._peak, ratio, money = _drawdown_observation(value, self._peak)
        return ratio, money

    def notify_fund(self, cash, value, fundvalue, shares):
        self._pending_value = self._observe_value(fundvalue if self._fundmode else value)

    def next(self):
        observation = self._pending_value
        self._pending_value = None
        if observation is None:
            broker = self.strategy.broker
            observation = self._observe_value(broker.fundvalue if self._fundmode else broker.getvalue())
        ratio, money = observation
        result = self.rets
        result.moneydown = money
        result.drawdown = -100.0 * ratio
        result.len = result.len + 1 if result.drawdown else 0
        result.max.moneydown = max(result.max.moneydown, result.moneydown)
        result.max.drawdown = max(result.max.drawdown, result.drawdown)
        result.max.len = max(result.max.len, result.len)
