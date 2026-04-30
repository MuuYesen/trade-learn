"""Legacy examine facade without vendored factor engines."""

from __future__ import annotations

from tradelearn.factor import FactorAnalyzer


class Examine:
    """Compatibility facade for old ``tradelearn.strategy.examine`` imports."""

    FactorAnalyzer = FactorAnalyzer


__all__ = ["Examine"]
