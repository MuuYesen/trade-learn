from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import pandas as pd


def select_top(
    scores: Mapping[Hashable, float],
    *,
    k: int,
    reverse: bool = True,
    min_score: float | None = None,
    max_score: float | None = None,
    exclude_nan: bool = True,
) -> list[Hashable]:
    """Return the top ``k`` keys by score.

    Parameters
    ----------
    scores:
        Mapping from asset identifier to numeric score.
    k:
        Number of identifiers to return.
    reverse:
        ``True`` selects highest scores first. ``False`` selects lowest scores.
    min_score:
        Optional lower score bound.
    max_score:
        Optional upper score bound.
    exclude_nan:
        Drop NaN values before ranking.
    """
    if k <= 0:
        return []

    items: list[tuple[Hashable, float]] = []
    for key, value in scores.items():
        score = float(value)
        if exclude_nan and math.isnan(score):
            continue
        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue
        items.append((key, score))

    return [
        key
        for key, _ in sorted(items, key=lambda item: item[1], reverse=reverse)[:k]
    ]


class TopKSelector:
    """Select the top-k symbols by score."""

    def __init__(
        self,
        k: int,
        *,
        ascending: bool = False,
        threshold: float | None = None,
    ) -> None:
        self.k = int(k)
        self.ascending = bool(ascending)
        self.threshold = threshold

    def select(self, scores: pd.Series) -> list[str]:
        """Return selected symbol labels."""
        return [
            str(symbol)
            for symbol in select_top(
                scores.to_dict(),
                k=self.k,
                reverse=not self.ascending,
                min_score=None if self.ascending else self.threshold,
                max_score=self.threshold if self.ascending else None,
            )
        ]

    def get_params(self) -> dict[str, Any]:
        """Return serializable selector parameters for tracking."""
        return {
            "type": type(self).__name__,
            "k": self.k,
            "ascending": self.ascending,
            "threshold": self.threshold,
        }


class EqualWeightOptimizer:
    """Build equal weights for selected symbols."""

    def __init__(self, gross: float = 1.0) -> None:
        self.gross = float(gross)

    def optimize(self, selected: Sequence[str], scores: pd.Series | None = None) -> pd.Series:
        """Return equal positive weights for selected symbols."""
        if not selected:
            return pd.Series(dtype="float64")
        weight = self.gross / len(selected)
        return pd.Series({str(symbol): weight for symbol in selected}, dtype="float64")

    def get_params(self) -> dict[str, Any]:
        """Return serializable optimizer parameters for tracking."""
        return {"type": type(self).__name__, "gross": self.gross}


class RiskPolicy:
    """Post-process portfolio weights with simple risk constraints."""

    def __init__(
        self,
        *,
        max_weight: float | None = None,
        min_abs_weight: float = 0.0,
        normalize: bool = False,
    ) -> None:
        self.max_weight = max_weight
        self.min_abs_weight = float(min_abs_weight)
        self.normalize = bool(normalize)

    def apply(self, weights: pd.Series) -> pd.Series:
        """Apply clipping, pruning, and optional normalization."""
        adjusted = weights.astype(float).copy()
        if self.max_weight is not None:
            cap = float(self.max_weight)
            adjusted = adjusted.clip(lower=-cap, upper=cap)
        if self.min_abs_weight > 0:
            adjusted = adjusted[adjusted.abs() >= self.min_abs_weight]
        if self.normalize and not adjusted.empty:
            gross = adjusted.abs().sum()
            if gross > 0:
                adjusted = adjusted / gross
        return adjusted

    def get_params(self) -> dict[str, Any]:
        """Return serializable risk parameters for tracking."""
        return {
            "type": type(self).__name__,
            "max_weight": self.max_weight,
            "min_abs_weight": self.min_abs_weight,
            "normalize": self.normalize,
        }


__all__ = [
    "EqualWeightOptimizer",
    "RiskPolicy",
    "TopKSelector",
    "select_top",
]
