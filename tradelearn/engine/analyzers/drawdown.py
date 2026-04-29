from __future__ import annotations

from ..analyzer import Analyzer


class Drawdown(Analyzer):
    """Calculates drawdowns (current, max, etc.)"""
    metric_key = "drawdown"

