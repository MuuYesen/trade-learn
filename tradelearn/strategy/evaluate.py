"""Legacy evaluate facade without vendored report engines."""

from __future__ import annotations

from tradelearn.report import Reporter


class Evaluate:
    """Compatibility facade for old ``tradelearn.strategy.evaluate`` imports."""

    Reporter = Reporter


__all__ = ["Evaluate"]
