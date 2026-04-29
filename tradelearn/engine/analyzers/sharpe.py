from __future__ import annotations

from ..analyzer import Analyzer


class SharpeRatio(Analyzer):
    """Calculates the Sharpe Ratio of the strategy."""
    metric_key = "sharpe"
    params = (("riskfreerate", 0.0),)

