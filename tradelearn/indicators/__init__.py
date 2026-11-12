"""Persistent indicator sources.

Choose talib (default source), tv or tdx explicitly. Retired PTA and duplicate
shortcut formulas are not imported, including through dynamic discovery.
"""

from __future__ import annotations

import importlib
from typing import Any

from tradelearn.indicators.base import FunctionIndicator, Indicator, IndicatorState

DEFAULT_SOURCE = "talib"
__all__ = ["DEFAULT_SOURCE", "FunctionIndicator", "Indicator", "IndicatorState"]


def __getattr__(name: str) -> Any:
    if name in {"talib", "tdx", "tv"}:
        return importlib.import_module(f"tradelearn.indicators.{name}")
    raise AttributeError(
        f"tradelearn.indicators has no {name!r}; use a qualified talib/tv/tdx source"
    )
