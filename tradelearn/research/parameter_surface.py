"""Summarize stable regions on a prespecified parameter grid."""

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import pandas as pd

Cell = tuple[Hashable, ...]


@dataclass(frozen=True)
class ParameterRegion:
    """Training-grid diagnostics. Cells in different components remain separate."""

    smoothed: pd.Series
    best_cell: Cell
    components: tuple[tuple[Cell, ...], ...]
    boundary_cells: tuple[Cell, ...]
    coverage: float
    unavailable_cells: int


def stable_region(
    scores: pd.Series,
    axes: Mapping[str, Sequence[Hashable]],
    *,
    standard_error: float,
) -> ParameterRegion:
    """Group neighboring cells within one supplied SE of the best smoothed cell.

    Axis values are in prespecified neighborhood order. Missing and nonfinite
    scores do not contribute to smoothing and are counted as unavailable.
    Higher scores are better. The supplied SE must apply to the prespecified
    selection metric. This calculation uses the supplied grid only and does
    not select one component or combine positions.
    """
    if not math.isfinite(standard_error) or standard_error < 0:
        raise ValueError("standard_error must be finite and nonnegative")
    if not axes or list(axes) != list(scores.index.names) or not scores.index.is_unique:
        raise ValueError("score index must be unique with names matching the ordered axes")
    positions = []
    for values in axes.values():
        labels = pd.Index(values, tupleize_cols=False)
        if labels.empty or not labels.is_unique or labels.hasnans:
            raise ValueError("axis values must be nonempty, unique, and nonmissing")
        positions.append({label: index for index, label in enumerate(labels)})

    cells: dict[tuple[int, ...], tuple[Cell, float, int]] = {}
    multi = isinstance(scores.index, pd.MultiIndex)
    for row, (label, score) in enumerate(scores.items()):
        cell = label if multi else (label,)
        try:
            point = tuple(axis[value] for axis, value in zip(positions, cell, strict=True))
        except KeyError as exc:
            raise ValueError(f"score label is outside the supplied axes: {cell!r}") from exc
        if not pd.isna(score) and math.isfinite(float(score)):
            cells[point] = (cell, float(score), row)
    if not cells:
        raise ValueError("no finite grid scores")

    smoothed = {}
    for point in cells:
        neighbors = product(*(range(x - 1, x + 2) for x in point))
        nearby_scores = [cells[neighbor][1] for neighbor in neighbors if neighbor in cells]
        smoothed[point] = sum(nearby_scores) / len(nearby_scores)
    best = min(smoothed, key=lambda point: (-smoothed[point], point))
    selected = {p for p, value in smoothed.items() if value >= smoothed[best] - standard_error}
    boundary = tuple(
        cells[p][0] for p in sorted(selected)
        if any(i == 0 or i == len(axis) - 1 for i, axis in zip(p, positions, strict=True))
    )

    components = []
    remaining = selected.copy()
    for point in sorted(selected):
        if point not in remaining:
            continue
        remaining.remove(point)
        stack, component = [point], []
        while stack:
            current = stack.pop()
            component.append(current)
            for axis, value in enumerate(current):
                for change in (-1, 1):
                    neighbor = current[:axis] + (value + change,) + current[axis + 1:]
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        components.append(tuple(cells[p][0] for p in sorted(component)))
    return ParameterRegion(
        smoothed=pd.Series(
            list(smoothed.values()),
            index=scores.index.take([entry[2] for entry in cells.values()]),
            name="smoothed",
        ),
        best_cell=cells[best][0],
        components=tuple(components),
        boundary_cells=boundary,
        coverage=len(selected) / len(cells),
        unavailable_cells=math.prod(map(len, positions)) - len(cells),
    )
