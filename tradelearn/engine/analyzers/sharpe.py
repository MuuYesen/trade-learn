"""Constant-state BT observed-value Sharpe profile; not the public TWR formula."""
from __future__ import annotations

import math

import pandas as pd

from ..analyzer import Analyzer
from ..base import TimeFrame
from .returns import _YEAR_PERIODS, _period_key, _validate_period


class SharpeRatio(Analyzer):
    """Accumulate completed observed-value periods without retaining full history."""

    params = (("timeframe", TimeFrame.Years), ("compression", 1),
              ("riskfreerate", 0.01), ("factor", None), ("convertrate", True),
              ("annualize", False), ("stddev_sample", False),
              ("daysfactor", None), ("legacyannual", False), ("fund", None))
    _metric_key = "sharpe"
    _is_streaming = True
    _requires_account_value = True

    def create_analysis(self):
        super().create_analysis()
        self._observation_started = False

    def start(self):
        if self.p.legacyannual:
            raise NotImplementedError(
                "legacyannual Sharpe historical observer profile is not implemented")
        self.timeframe = self.p.timeframe or self.data._timeframe
        self.compression = self.p.compression or self.data._compression
        _validate_period(self.timeframe, self.compression)
        self._fundmode = (getattr(self.strategy.broker, "fundmode", False)
                          if self.p.fund is None else self.p.fund)
        if self._fundmode and not hasattr(self.strategy.broker, "fundvalue"):
            raise NotImplementedError("fund-mode Sharpe requires authoritative broker fundvalue")
        self._factor = (self.p.daysfactor if self.p.timeframe == TimeFrame.Days
                        and self.p.daysfactor is not None else self.p.factor)
        if self._factor is None:
            self._factor = _YEAR_PERIODS.get(self.p.timeframe)
        self._rate = self.p.riskfreerate
        if self._factor is not None and self.p.convertrate:
            self._rate = (1 + self._rate) ** (1 / self._factor) - 1
        self._anchor = self._value()
        self._last_value = self._anchor
        self._last_period = None
        self._count = 0
        self._mean = self._m2 = 0.0
        self._observation_started = True

    def _value(self):
        broker = self.strategy.broker
        return broker.fundvalue if self._fundmode else broker.getvalue()

    def _commit_period(self):
        value = self._last_value / self._anchor - 1
        if self._factor is not None and not self.p.convertrate:
            value = (1 + value) ** self._factor - 1
        excess = value - self._rate
        self._count += 1
        delta = excess - self._mean
        self._mean += delta / self._count
        self._m2 += delta * (excess - self._mean)

    def _before_business(self):
        self._sampled_value = self._value()

    def _before_next(self, phase):
        context = getattr(getattr(self.strategy, "_runtime", None), "context", None)
        timestamp = (pd.Timestamp(context.ts_event_ns, unit="ns", tz="UTC") if context is not None
                     else self.strategy.datetime.datetime())
        key = _period_key(timestamp, self.timeframe, self.compression)
        if self._last_period is not None and key > self._last_period:
            self._commit_period()
            self._anchor = self._last_value
        self._last_period = key
        self._last_value = self._sampled_value
        return True

    def stop(self):
        if not self._observation_started:
            return
        self._observation_started = False
        if self._last_period is not None:
            self._commit_period()
        denominator = self._count - self.p.stddev_sample
        ratio = None
        if denominator > 0 and self._m2 > 0:
            ratio = self._mean / math.sqrt(self._m2 / denominator)
            if self._factor is not None and self.p.convertrate and self.p.annualize:
                ratio *= math.sqrt(self._factor)
        self.ratio = self.rets["sharperatio"] = ratio
