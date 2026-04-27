from __future__ import annotations
from typing import Any, Dict
from tradelearn.backtest.analyzer import Analyzer

class Returns(Analyzer):
    """Calculates returns (total, average, etc.)"""
    metric_key = "returns"

