"""Shared parameter-grid helpers."""

from __future__ import annotations

from itertools import product
from typing import Any


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of parameter dictionaries."""

    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, item, strict=True)) for item in product(*values)]
