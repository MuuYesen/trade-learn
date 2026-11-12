"""BT observed-value log-return profile, distinct from public cashflow/TWR metrics."""
from __future__ import annotations

import math

import pandas as pd

from ..analyzer import Analyzer
from ..base import TimeFrame

_YEAR_PERIODS = {TimeFrame.Days: 252.0, TimeFrame.Weeks: 52.0,
                 TimeFrame.Months: 12.0, TimeFrame.Years: 1.0}


def _period_key(timestamp, timeframe, compression):
    """Compare real observed calendar buckets; do not infer a sampling cadence."""
    if timeframe == TimeFrame.NoTimeFrame:
        return (0,)
    if timeframe == TimeFrame.Years:
        return (timestamp.year,)
    if timeframe == TimeFrame.Months:
        return timestamp.year, timestamp.month
    if timeframe == TimeFrame.Weeks:
        return tuple(timestamp.isocalendar()[:2])
    if timeframe == TimeFrame.Days:
        return timestamp.year, timestamp.month, timestamp.day
    units = {TimeFrame.Minutes: 60_000_000, TimeFrame.Seconds: 1_000_000,
             TimeFrame.MicroSeconds: 1}
    if timeframe not in units:
        raise ValueError("a time-based analyzer timeframe is required")
    elapsed = ((timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second)
               * 1_000_000 + timestamp.microsecond)
    return timestamp.date().toordinal(), elapsed // (units[timeframe] * compression)


def _validate_period(timeframe, compression):
    if timeframe not in tuple(range(TimeFrame.MicroSeconds, TimeFrame.NoTimeFrame + 1)):
        raise ValueError("a declared time-based analyzer timeframe is required")
    if timeframe < TimeFrame.Days:
        if compression is None or not math.isfinite(float(compression)) or compression <= 0:
            raise ValueError("a positive declared subday compression is required")


class Returns(Analyzer):
    """Constant-state BT Returns; result is finalized at stop, as in BT.

    These are logarithmic returns on observed broker value, without cashflow
    adjustment. Annual defaults belong to this explicit BT compatibility profile.
    """

    params = (("timeframe", None), ("compression", None), ("_doprenext", True),
              ("tann", None), ("fund", None))
    _metric_key = "returns"
    _is_streaming = True
    _requires_account_value = True

    def create_analysis(self):
        super().create_analysis()
        self._observation_started = False

    def start(self):
        self.timeframe = self.p.timeframe or self.data._timeframe
        self.compression = self.p.compression or self.data._compression
        _validate_period(self.timeframe, self.compression)
        self._fundmode = (getattr(self.strategy.broker, "fundmode", False)
                          if self.p.fund is None else self.p.fund)
        if self._fundmode and not hasattr(self.strategy.broker, "fundvalue"):
            raise NotImplementedError("fund-mode Returns requires authoritative broker fundvalue")
        self._initial_value = self._value()
        self._period_count = 0
        self._last_period = None
        self._observation_started = True

    def _value(self):
        broker = self.strategy.broker
        return broker.fundvalue if self._fundmode else broker.getvalue()

    def _before_next(self, phase):
        runtime = getattr(self.strategy, "_runtime", None)
        context = getattr(runtime, "context", None)
        timestamp = (pd.Timestamp(context.ts_event_ns, unit="ns", tz="UTC") if context is not None
                     else self.strategy.datetime.datetime())
        key = _period_key(timestamp, self.timeframe, self.compression)
        if self._last_period is None or key > self._last_period:
            self._period_count += 1
            self._last_period = key
        return phase != "prenext" or self.p._doprenext

    def stop(self):
        if not self._observation_started:
            return
        end = self._value()
        growth = end / self._initial_value if self._initial_value else -1.0
        total = math.log(growth) if growth >= 0 else float("-inf")
        average = total / self._period_count
        annual_periods = self.p.tann or _YEAR_PERIODS.get(self.timeframe)
        if annual_periods is None:
            annual_periods = _YEAR_PERIODS.get(self.data._timeframe, 1.0)
        normalized = math.expm1(average * annual_periods) if average > -math.inf else average
        self.rets.update(rtot=total, ravg=average, rnorm=normalized, rnorm100=normalized * 100)
