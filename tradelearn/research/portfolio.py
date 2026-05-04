from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from tradelearn.research.run import current_run, suspend_tracking, tracked


@runtime_checkable
class Selector(Protocol):
    """Protocol for user-defined score-to-selection components."""

    def select(self, scores: Mapping[Hashable, float] | pd.Series) -> Any:
        """Return selected symbols or a panel selection mask."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serializable parameters for tracking."""
        ...


@runtime_checkable
class Weighter(Protocol):
    """Protocol for user-defined selection-to-weight components."""

    def allocate(self, selected: Any) -> pd.Series:
        """Return portfolio weights for selected symbols."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serializable parameters for tracking."""
        ...


@runtime_checkable
class Constraint(Protocol):
    """Protocol for user-defined weight post-processors."""

    def apply(self, weights: Mapping[Hashable, float] | pd.Series) -> pd.Series:
        """Return constrained weights."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serializable parameters for tracking."""
        ...


@tracked(category="portfolio")
def select_top(
    scores: Mapping[Hashable, float],
    *,
    k: int,
    reverse: bool = True,
    min_score: float | None = None,
    max_score: float | None = None,
    exclude_nan: bool = True,
) -> list[Hashable] | pd.Series:
    """Return top-k symbols, or a panel selection mask for panel scores."""
    if _is_panel_series(scores):
        return _select_top_panel(
            pd.Series(scores),
            k=k,
            reverse=reverse,
            min_score=min_score,
            max_score=max_score,
            exclude_nan=exclude_nan,
        )
    return _select_top_single(
        scores,
        k=k,
        reverse=reverse,
        min_score=min_score,
        max_score=max_score,
        exclude_nan=exclude_nan,
    )


def _select_top_single(
    scores: Mapping[Hashable, float],
    *,
    k: int,
    reverse: bool = True,
    min_score: float | None = None,
    max_score: float | None = None,
    exclude_nan: bool = True,
) -> list[Hashable]:
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


def _select_top_panel(
    scores: pd.Series,
    *,
    k: int,
    reverse: bool,
    min_score: float | None,
    max_score: float | None,
    exclude_nan: bool,
) -> pd.Series:
    if k <= 0:
        return _empty_panel_series("selected", dtype="bool")

    parts: list[pd.Series] = []
    time_level = _time_level(scores.index)
    for timestamp, daily_scores in scores.groupby(level=time_level):
        selected = _select_top_single(
            daily_scores.droplevel(time_level),
            k=k,
            reverse=reverse,
            min_score=min_score,
            max_score=max_score,
            exclude_nan=exclude_nan,
        )
        if not selected:
            continue
        index = pd.MultiIndex.from_product(
            [[timestamp], [str(symbol) for symbol in selected]],
            names=["timestamp", "symbol"],
        )
        parts.append(pd.Series(True, index=index, name="selected", dtype="bool"))
    if not parts:
        return _empty_panel_series("selected", dtype="bool")
    return pd.concat(parts).sort_index().rename("selected")


@tracked(category="portfolio")
def equal_weight(
    selected: Sequence[Hashable] | pd.Series,
    *,
    gross: float = 1.0,
) -> pd.Series:
    """Return equal positive weights for selected symbols or panel selections."""
    if _is_panel_series(selected):
        return _equal_weight_panel(pd.Series(selected), gross=gross)
    if len(selected) == 0:
        return pd.Series(dtype="float64")
    weight = float(gross) / len(selected)
    return pd.Series({str(symbol): weight for symbol in selected}, dtype="float64")


def _equal_weight_panel(selected: pd.Series, *, gross: float = 1.0) -> pd.Series:
    if selected.empty:
        return _empty_panel_series("weight")

    selected = selected[selected.astype(bool)]
    if selected.empty:
        return _empty_panel_series("weight")

    parts: list[pd.Series] = []
    time_level = _time_level(selected.index)
    for timestamp, daily_selected in selected.groupby(level=time_level):
        symbols = daily_selected.index.droplevel(time_level).astype(str)
        if symbols.empty:
            continue
        weight = float(gross) / len(symbols)
        index = pd.MultiIndex.from_product(
            [[timestamp], symbols],
            names=["timestamp", "symbol"],
        )
        parts.append(pd.Series(weight, index=index, name="weight", dtype="float64"))
    if not parts:
        return _empty_panel_series("weight")
    return pd.concat(parts).sort_index().rename("weight")


@tracked(category="portfolio")
def apply_constraints(
    weights: Mapping[Hashable, float] | pd.Series,
    *,
    max_weight: float | None = None,
    min_abs_weight: float = 0.0,
    normalize: bool = False,
) -> pd.Series:
    """Apply clipping, pruning, and optional gross normalization."""
    if _is_panel_series(weights):
        return _apply_constraints_panel(
            pd.Series(weights),
            max_weight=max_weight,
            min_abs_weight=min_abs_weight,
            normalize=normalize,
        )
    return _apply_constraints_single(
        weights,
        max_weight=max_weight,
        min_abs_weight=min_abs_weight,
        normalize=normalize,
    )


def _apply_constraints_single(
    weights: Mapping[Hashable, float] | pd.Series,
    *,
    max_weight: float | None,
    min_abs_weight: float,
    normalize: bool,
) -> pd.Series:
    adjusted = pd.Series(weights, dtype="float64").copy()
    adjusted.index = adjusted.index.astype(str)
    if max_weight is not None:
        cap = float(max_weight)
        adjusted = adjusted.clip(lower=-cap, upper=cap)
    if min_abs_weight > 0:
        adjusted = adjusted[adjusted.abs() >= float(min_abs_weight)]
    if normalize and not adjusted.empty:
        gross = adjusted.abs().sum()
        if gross > 0:
            adjusted = adjusted / gross
    return adjusted


def _apply_constraints_panel(
    weights: pd.Series,
    *,
    max_weight: float | None,
    min_abs_weight: float,
    normalize: bool,
) -> pd.Series:
    if weights.empty:
        return _empty_panel_series("weight")

    parts: list[pd.Series] = []
    time_level = _time_level(weights.index)
    for timestamp, daily_weights in weights.groupby(level=time_level):
        adjusted = _apply_constraints_single(
            daily_weights.droplevel(time_level),
            max_weight=max_weight,
            min_abs_weight=min_abs_weight,
            normalize=normalize,
        )
        if adjusted.empty:
            continue
        index = pd.MultiIndex.from_product(
            [[timestamp], adjusted.index.astype(str)],
            names=["timestamp", "symbol"],
        )
        parts.append(pd.Series(adjusted.to_numpy(), index=index, name="weight"))
    if not parts:
        return _empty_panel_series("weight")
    return pd.concat(parts).sort_index().astype("float64").rename("weight")


@tracked(category="portfolio")
def topk_equal_weights(
    scores: pd.Series,
    *,
    k: int,
    gross: float = 1.0,
    max_weight: float | None = None,
    min_abs_weight: float = 0.0,
    normalize: bool = False,
) -> pd.Series:
    """Build MultiIndex(timestamp, symbol) equal weights from panel scores."""
    with suspend_tracking():
        selected = select_top(scores, k=k)
        weights = equal_weight(selected, gross=gross)
        return apply_constraints(
            weights,
            max_weight=max_weight,
            min_abs_weight=min_abs_weight,
            normalize=normalize,
        )


def _is_panel_series(value: object) -> bool:
    return (
        isinstance(value, pd.Series)
        and isinstance(value.index, pd.MultiIndex)
        and value.index.nlevels >= 2
    )


def _time_level(index: pd.MultiIndex) -> str | int:
    return index.names[0] if index.names[0] is not None else 0


def _empty_panel_series(name: str, *, dtype: str = "float64") -> pd.Series:
    return pd.Series(
        dtype=dtype,
        index=pd.MultiIndex.from_arrays([[], []], names=["timestamp", "symbol"]),
        name=name,
    )


class TopK:
    """Object interface for top-k portfolio selection."""

    def __init__(self, k: int, *, reverse: bool = True) -> None:
        self.k = int(k)
        self.reverse = bool(reverse)

    def select(self, scores: Mapping[Hashable, float] | pd.Series) -> list[Hashable] | pd.Series:
        """Return selected symbols or a panel selection mask."""
        return select_top(scores, k=self.k, reverse=self.reverse)

    def get_params(self) -> dict[str, Any]:
        """Return serializable selector parameters."""
        return {"type": type(self).__name__, "k": self.k, "reverse": self.reverse}


class EqualWeight:
    """Object interface for equal-weight allocation."""

    def __init__(self, *, gross: float = 1.0) -> None:
        self.gross = float(gross)

    def allocate(self, selected: Sequence[Hashable] | pd.Series) -> pd.Series:
        """Return equal weights for selected symbols."""
        return equal_weight(selected, gross=self.gross)

    def get_params(self) -> dict[str, Any]:
        """Return serializable allocation parameters."""
        return {"type": type(self).__name__, "gross": self.gross}


class Constraints:
    """Object interface for portfolio weight constraints."""

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

    def apply(self, weights: Mapping[Hashable, float] | pd.Series) -> pd.Series:
        """Apply clipping, pruning, and optional gross normalization."""
        return apply_constraints(
            weights,
            max_weight=self.max_weight,
            min_abs_weight=self.min_abs_weight,
            normalize=self.normalize,
        )

    def get_params(self) -> dict[str, Any]:
        """Return serializable constraint parameters."""
        return {
            "type": type(self).__name__,
            "max_weight": self.max_weight,
            "min_abs_weight": self.min_abs_weight,
            "normalize": self.normalize,
        }


class Allocator:
    """Build portfolio weights from scores with object-style components."""

    def __init__(
        self,
        *,
        select: Selector,
        weight: Weighter,
        constrain: Constraint | None = None,
    ) -> None:
        self.select = select
        self.weight = weight
        self.constrain = constrain

    def build(self, scores: Mapping[Hashable, float] | pd.Series) -> pd.Series:
        """Build target weights from scores."""
        _record_allocator_step(self)
        with suspend_tracking():
            selected = self.select.select(scores)
            weights = self.weight.allocate(selected)
            if self.constrain is not None:
                weights = self.constrain.apply(weights)
        return pd.Series(weights, dtype="float64", name="weight")

    def get_params(self) -> dict[str, Any]:
        """Return flat serializable builder parameters for tracking."""
        params: dict[str, Any] = {}
        params.update(_component_params("select", self.select))
        params.update(_component_params("weight", self.weight))
        if self.constrain is not None:
            params.update(_component_params("constrain", self.constrain))
        return params


def _component_params(prefix: str, component: Any) -> dict[str, Any]:
    raw = (
        component.get_params()
        if hasattr(component, "get_params")
        else {"type": type(component).__name__}
    )
    return {f"{prefix}.{key}": value for key, value in raw.items()}


def _record_allocator_step(builder: Allocator) -> None:
    run = current_run()
    if run is not None:
        run.record_step("Allocator", "portfolio", builder.get_params())


class RiskfolioOptimizer:
    """Optional Riskfolio-Lib portfolio optimizer wrapper."""

    def __init__(
        self,
        *,
        model: str = "Classic",
        rm: str = "MV",
        obj: str = "Sharpe",
        rf: float = 0.0,
        risk_aversion: float = 0.0,
        hist: bool = True,
        method_mu: str = "hist",
        method_cov: str = "hist",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.rm = rm
        self.obj = obj
        self.rf = float(rf)
        self.risk_aversion = float(kwargs.pop("l", risk_aversion))
        self.hist = bool(hist)
        self.method_mu = method_mu
        self.method_cov = method_cov
        self.kwargs = dict(kwargs)

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Return optimized target weights from a returns matrix."""
        riskfolio = _import_optional(
            "riskfolio",
            package="Riskfolio-Lib",
        )
        portfolio = riskfolio.Portfolio(returns=returns.astype("float64"))
        portfolio.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov)
        result = portfolio.optimization(
            model=self.model,
            rm=self.rm,
            obj=self.obj,
            rf=self.rf,
            l=self.risk_aversion,
            hist=self.hist,
            **self.kwargs,
        )
        return _weights_series(result)

    def get_params(self) -> dict[str, Any]:
        """Return serializable optimizer parameters for tracking."""
        params = {
            "type": type(self).__name__,
            "model": self.model,
            "rm": self.rm,
            "obj": self.obj,
            "rf": self.rf,
            "risk_aversion": self.risk_aversion,
            "hist": self.hist,
            "method_mu": self.method_mu,
            "method_cov": self.method_cov,
        }
        params.update(self.kwargs)
        return params


def _import_optional(module: str, *, package: str):
    try:
        imported = import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{package} is required for this feature. Install with "
            f"`pip install trade-learn[all]` or `pip install {package}`."
        ) from exc
    if imported is None:
        raise ImportError(
            f"{package} is required for this feature. Install with "
            f"`pip install trade-learn[all]` or `pip install {package}`."
        )
    return imported


def _weights_series(result: Any) -> pd.Series:
    if isinstance(result, pd.DataFrame):
        if "weights" in result.columns:
            series = result["weights"]
        elif "weight" in result.columns:
            series = result["weight"]
        elif result.shape[1] == 1:
            series = result.iloc[:, 0]
        else:
            raise ValueError("Riskfolio result DataFrame must contain one weight column")
    elif isinstance(result, pd.Series):
        series = result
    elif isinstance(result, Mapping):
        series = pd.Series(result)
    else:
        raise TypeError("Riskfolio optimizer returned unsupported weight output")
    series = series.astype("float64")
    series.index = series.index.astype(str)
    series.name = "weight"
    return series


__all__ = [
    "Constraint",
    "Constraints",
    "EqualWeight",
    "RiskfolioOptimizer",
    "Selector",
    "TopK",
    "Allocator",
    "Weighter",
    "apply_constraints",
    "equal_weight",
    "select_top",
    "topk_equal_weights",
]
