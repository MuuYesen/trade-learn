"""trade-learn public package namespace."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.2.0"

__all__ = ["__version__", "ta"]


def __getattr__(name: str) -> Any:
    """Lazily expose public namespace aliases."""
    if name == "ta":
        from tradelearn import indicators

        return indicators
    if name == "lab":
        from tradelearn import lab

        return lab
    if name == "cli":
        from tradelearn import cli

        return cli
    raise AttributeError(f"module 'tradelearn' has no attribute {name!r}")
