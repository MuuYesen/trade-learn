"""Shared annualization periods for backtest summaries, reports and artifacts.

A single resolver decides how many return observations make a year, so the
engine summary, :class:`~tradelearn.report.Reporter`, report artifacts and
MLflow tags all describe the same number with the same source label.

Resolution order:

1. an explicit, validated ``periods`` supplied by the user (``"explicit"``);
2. the observed sampling rate of an intraday return index covering at least
   one full week (``"inferred_return_frequency"``);
3. 252 for intraday data covering less than a week
   (``"insufficient_intraday_coverage"`` -- a compatibility fallback, not a
   valid market annualization; pass ``periods`` explicitly);
4. the 252 per-bar profile for daily or slower data, or when the index cannot
   be measured (``"default_per_bar_profile"``).

Daily data is never promoted to 365 automatically: whether a market trades
every calendar day is a calendar fact the return index cannot prove, so 7x24
daily series must pass ``periods=365`` explicitly.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PERIODS = 252
SOURCE_EXPLICIT = "explicit"
SOURCE_INFERRED = "inferred_return_frequency"
SOURCE_DEFAULT = "default_per_bar_profile"
SOURCE_INSUFFICIENT = "insufficient_intraday_coverage"
MIN_INFERENCE_SPAN_SECONDS = 7 * 24 * 3600
ANNUALIZED_SUMMARY_METRICS = ("annual_return", "volatility", "sharpe", "sortino", "calmar")


def validate_periods(periods: Any) -> int | None:
    """Return a positive integer periods-per-year value, or ``None`` when unset.

    Shares the metric layer's positive-integer rule. Integral floats such as
    ``365.0`` (accepted by older Reporter callers) are converted to ``int``;
    fractional, non-finite, boolean and non-numeric values are rejected.
    """
    from tradelearn.metrics._common import validate_periods as validate_metric_periods

    if periods is None:
        return None
    if isinstance(periods, (bool, np.bool_)) or not isinstance(periods, numbers.Real):
        raise ValueError("periods must be a positive integer")
    if not isinstance(periods, numbers.Integral):
        value = float(periods)
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("periods must be a positive integer")
        periods = int(value)
    periods = int(periods)
    validate_metric_periods(periods)
    return periods


def _measure(returns: Any) -> tuple[int | None, bool]:
    """Return ``(periods, intraday)`` measured from a clean intraday index.

    Counting observations against the calendar span they cover works for
    intraday data without assuming a session length. The result is a measured
    observation density, not a verified trading calendar. It is only trusted
    once the index covers at least one full week, so a single session (e.g.
    390 one-minute bars) is not extrapolated to 525,600 bars a year and
    weekends/overnight gaps are included in the measurement. Indexes with NaT,
    duplicates or decreasing timestamps are not measured. Daily and slower
    indexes are never measured, so 7x24 daily data is not guessed as 365.
    """
    try:
        index = getattr(returns, "index", None)
        if index is None or len(index) < 3:
            return None, False
        values = pd.DatetimeIndex(index)
        if values.hasnans or not values.is_monotonic_increasing or not values.is_unique:
            return None, False
        span = (values[-1] - values[0]).total_seconds()
        if not np.isfinite(span) or span <= 0:
            return None, False
        step = float(np.median(np.diff(values.as_unit("ns").asi8))) / 1e9
        if not np.isfinite(step) or step <= 0 or step >= 24 * 3600:
            return None, False
        if span < MIN_INFERENCE_SPAN_SECONDS:
            return None, True
        per_year = (len(values) - 1) * (365.0 * 24 * 3600) / span
        if not np.isfinite(per_year) or per_year <= 0:
            return None, True
        return int(round(per_year)), True
    except (TypeError, ValueError):
        return None, False


def _return_frequency(returns: Any) -> int | None:
    """Measured periods-per-year for a week-covering intraday index, else ``None``."""
    return _measure(returns)[0]


def infer_periods(returns: Any, fallback: int = DEFAULT_PERIODS) -> int:
    """Infer periods-per-year from the return index, falling back otherwise."""
    inferred = _return_frequency(returns)
    return fallback if inferred is None else inferred


def resolve_periods(returns: Any, periods: Any = None) -> tuple[int, str]:
    """Return ``(periods, period_source)`` for a return series.

    ``insufficient_intraday_coverage`` keeps the backward-compatible 252 value
    but marks it as not a valid market annualization; pass ``periods``.
    """
    explicit = validate_periods(periods)
    if explicit is not None:
        return explicit, SOURCE_EXPLICIT
    inferred, intraday = _measure(returns)
    if inferred is not None:
        return inferred, SOURCE_INFERRED
    return DEFAULT_PERIODS, SOURCE_INSUFFICIENT if intraday else SOURCE_DEFAULT


def periods_from_metadata(metric_metadata: Any) -> tuple[int, str] | None:
    """Return resolved periods recorded by a producer, ignoring legacy defaults.

    Legacy artifacts labelled every summary with ``default_per_bar_profile``
    252 regardless of bar size, so that label is not trusted; consumers keep
    inferring from the returns in that case. Conflicting records also return
    ``None``. Returns ``(periods, period_source)``.
    """
    if not isinstance(metric_metadata, dict):
        return None
    found = set()
    for name in ANNUALIZED_SUMMARY_METRICS:
        record = metric_metadata.get(name)
        if not isinstance(record, dict):
            continue
        if record.get("period_source") not in (SOURCE_EXPLICIT, SOURCE_INFERRED):
            continue
        try:
            value = validate_periods(record.get("periods"))
        except (TypeError, ValueError):
            return None
        if value is not None:
            found.add((value, record["period_source"]))
    return found.pop() if len(found) == 1 else None
