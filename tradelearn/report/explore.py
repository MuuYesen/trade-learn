"""Interactive report exploration."""

from __future__ import annotations

from typing import Any


def explore_trades(trades: Any) -> Any:
    """Open pygwalker for a trades table."""
    try:
        import pygwalker as pyg
    except ImportError as exc:
        raise ImportError(
            "explore() requires pygwalker. Install with: pip install trade-learn[lab]"
        ) from exc
    if getattr(trades, "empty", False):
        raise ValueError("explore() requires at least one trade row")
    return pyg.walk(trades)
