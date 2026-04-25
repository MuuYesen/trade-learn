"""trade-learn public package namespace."""

from __future__ import annotations

from typing import Any

__all__ = ["ta"]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        from tradelearn import indicators

        return indicators
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
