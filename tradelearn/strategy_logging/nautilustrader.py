"""Explicit, optional logging strategy entry point for nautilustrader."""

import importlib

__all__ = ["TradeLogStrategy"]


def __getattr__(name: str):
    if name == "TradeLogStrategy":
        return importlib.import_module("leafquant_monitor.strategy").TradeLogStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"TradeLogStrategy"})
