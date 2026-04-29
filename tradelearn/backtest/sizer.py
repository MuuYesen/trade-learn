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


class FixedSize(Sizer):
    """Shared fixed-size sizer used by runtime facades."""

    def __init__(self, stake: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stake = stake

    def getsizing(self, data: Any, isbuy: bool, **kwargs: Any) -> float:
        return float(self.stake)
