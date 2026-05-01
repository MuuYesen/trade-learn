from __future__ import annotations

import math
from collections.abc import Hashable, Mapping


def select_top(
    scores: Mapping[Hashable, float],
    *,
    k: int,
    reverse: bool = True,
    min_score: float | None = None,
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
        items.append((key, score))

    return [
        key
        for key, _ in sorted(items, key=lambda item: item[1], reverse=reverse)[:k]
    ]


__all__ = ["select_top"]
