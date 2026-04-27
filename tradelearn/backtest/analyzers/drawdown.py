from __future__ import annotations
from typing import Any, Dict
from tradelearn.backtest.analyzer import Analyzer

class Drawdown(Analyzer):
    """Calculates drawdowns (current, max, etc.)"""
    metric_key = "drawdown"

