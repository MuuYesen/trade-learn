from __future__ import annotations

from typing import Any


class Sizer:
    """Minimalist sizer interface."""

    def __init__(self, **kwargs) -> None:
        self.strategy: Any = None
        self.broker: Any = None

    def _set(self, strategy: Any, broker: Any) -> None:
        self.strategy = strategy
        self.broker = broker

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        """Calculate the size for an order."""
        return 1.0
