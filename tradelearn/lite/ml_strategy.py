"""Lite ML strategy facade."""

from __future__ import annotations

from tradelearn.lite.strategy import Strategy
from tradelearn.ml.base import MLStrategyMixin


class MLStrategy(MLStrategyMixin, Strategy):
    """Lite-facing ML strategy base class using the shared ML runtime."""


__all__ = ["MLStrategy"]
