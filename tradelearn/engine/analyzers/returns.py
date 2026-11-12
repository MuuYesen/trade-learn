from __future__ import annotations

from ..analyzer import Analyzer


class Returns(Analyzer):
    """Calculates returns (total, average, etc.)"""
    _metric_key = "returns"

