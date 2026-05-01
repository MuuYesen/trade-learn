from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from importlib import import_module
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


class PortfolioConstraints:
    """Post-process portfolio weights with simple portfolio constraints."""

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
        """Return serializable constraint parameters for tracking."""
        return {
            "type": type(self).__name__,
            "max_weight": self.max_weight,
            "min_abs_weight": self.min_abs_weight,
            "normalize": self.normalize,
        }


RiskPolicy = PortfolioConstraints


class RiskfolioOptimizer:
    """Optional Riskfolio-Lib portfolio optimizer wrapper.

    The wrapper turns a return matrix into target weights. It does not submit
    orders or change backtest execution semantics.
    """

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
            extra="riskfolio",
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


def _import_optional(module: str, *, extra: str, package: str):
    try:
        imported = import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{package} is required for this feature. Install with "
            f"`pip install trade-learn[{extra}]`."
        ) from exc
    if imported is None:
        raise ImportError(
            f"{package} is required for this feature. Install with "
            f"`pip install trade-learn[{extra}]`."
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
    "EqualWeightOptimizer",
    "PortfolioConstraints",
    "RiskPolicy",
    "RiskfolioOptimizer",
    "TopKSelector",
    "select_top",
]
