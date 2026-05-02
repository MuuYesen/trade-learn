from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import pandas as pd

_CURRENT_RUN: contextvars.ContextVar[ResearchRun | None] = contextvars.ContextVar(
    "tradelearn_research_run",
    default=None,
)


@dataclass(frozen=True)
class ResearchStep:
    """A recorded research operation."""

    name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly step payload."""
        return {
            "name": self.name,
            "category": self.category,
            "params": _json_safe(self.params),
        }


@dataclass
class ResearchResult:
    """Structured output of a research workflow."""

    name: str
    steps: list[ResearchStep] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    features: Any | None = None
    target: Any | None = None
    selected_features: Sequence[str] | None = None
    model: Any | None = None
    scores: pd.Series | None = None
    selected: Sequence[Any] | None = None
    weights: pd.Series | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def as_weight_dict(self) -> dict[str, float]:
        """Return weights as a plain dict suitable for target_weights()."""
        if self.weights is None:
            return {}
        return {str(symbol): float(weight) for symbol, weight in self.weights.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly research payload."""
        payload: dict[str, Any] = {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "params": _json_safe(self.params),
        }
        result: dict[str, Any] = {}
        if self.scores is not None:
            result["scores"] = _series_dict(self.scores)
        if self.selected is not None:
            result["selected"] = [str(item) for item in self.selected]
        if self.weights is not None:
            result["weights"] = _series_dict(self.weights)
        if result:
            payload["result"] = result
        if self.selected_features is not None:
            payload["selected_features"] = [str(item) for item in self.selected_features]
        if self.model is not None:
            payload["model"] = type(self.model).__name__
        if self.artifacts:
            payload["artifacts"] = _json_safe(self.artifacts)
        return payload


class BoundResearchResult:
    """Strategy-bound research result view."""

    def __init__(self, result: ResearchResult, strategy: Any) -> None:
        self._result = result
        self._strategy = strategy
        self.weights = ResearchWeights(result.weights, strategy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    @property
    def raw(self) -> ResearchResult:
        """Return the original research result."""
        return self._result

    def as_weight_dict(self) -> dict[str, float]:
        """Return current weights as a plain dict suitable for target_weights()."""
        current = self.weights[0]
        if current is None:
            return {}
        return {str(symbol): float(weight) for symbol, weight in current.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly research payload."""
        return self._result.to_dict()


class ResearchWeights:
    """Strategy-bound weights supporting TradeLearn ``[0]`` current-bar access."""

    def __init__(self, raw: Any, strategy: Any) -> None:
        self.raw = raw
        self._strategy = strategy

    def __getitem__(self, offset: int) -> Any:
        if not isinstance(offset, int):
            return self.raw[offset]
        return self._slice_for_offset(offset)

    def items(self):
        current = self[0]
        return current.items() if hasattr(current, "items") else dict(current).items()

    def to_dict(self) -> dict[str, float]:
        return {str(symbol): float(weight) for symbol, weight in self.items()}

    def _slice_for_offset(self, offset: int) -> Any:
        weights = self.raw
        if weights is None:
            return pd.Series(dtype="float64", name="weight")
        if isinstance(weights, pd.Series):
            if isinstance(weights.index, pd.MultiIndex) and weights.index.nlevels >= 2:
                return _slice_multiindex_weights(
                    weights,
                    _strategy_timestamp(self._strategy),
                    offset,
                )
            raise ValueError(
                "research_result.weights[0] requires MultiIndex(timestamp, symbol) weights. "
                "For static weights, call target_weights(weights) directly."
            )
        if isinstance(weights, pd.DataFrame):
            if {"timestamp", "symbol", "weight"}.issubset(weights.columns):
                return _slice_long_weights(weights, _strategy_timestamp(self._strategy), offset)
            return _slice_wide_weights(weights, _strategy_timestamp(self._strategy), offset)
        if isinstance(weights, dict):
            raise ValueError(
                "research_result.weights[0] requires MultiIndex(timestamp, symbol) weights. "
                "For static weights, call target_weights(weights) directly."
            )
        return weights


def bind_research_result(result: Any, strategy: Any) -> Any:
    """Return a strategy-bound research result view when possible."""

    if isinstance(result, BoundResearchResult):
        return result
    if isinstance(result, ResearchResult):
        return BoundResearchResult(result, strategy)
    return result


class ResearchRun:
    """Context manager that records tracked research function calls."""

    def __init__(self, name: str) -> None:
        self.name = str(name)
        self.steps: list[ResearchStep] = []
        self._token: contextvars.Token[ResearchRun | None] | None = None

    def __enter__(self) -> ResearchRun:
        self._token = _CURRENT_RUN.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _CURRENT_RUN.reset(self._token)
            self._token = None

    def record_step(self, name: str, category: str, params: dict[str, Any]) -> None:
        """Append a recorded step."""
        self.steps.append(ResearchStep(name=name, category=category, params=params))

    def finish(
        self,
        *,
        features: Any | None = None,
        target: Any | None = None,
        selected_features: Sequence[str] | None = None,
        model: Any | None = None,
        scores: Any | None = None,
        selected: Sequence[Any] | None = None,
        weights: Any | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> ResearchResult:
        """Build the final structured result."""
        return ResearchResult(
            name=self.name,
            steps=list(self.steps),
            params=_flatten_step_params(self.steps),
            features=features,
            target=target,
            selected_features=selected_features,
            model=model,
            scores=(
                None
                if scores is None
                else pd.Series(scores, name=getattr(scores, "name", "score"))
            ),
            selected=None if selected is None else [str(item) for item in selected],
            weights=None if weights is None else pd.Series(weights, dtype="float64", name="weight"),
            artifacts=dict(artifacts or {}),
        )

    class preprocess:
        """Static-method convenience facade for tracked preprocess functions."""

        @staticmethod
        def fill_missing(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import fill_missing

            return fill_missing(*args, **kwargs)

        @staticmethod
        def fill_by_group(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import fill_by_group

            return fill_by_group(*args, **kwargs)

        @staticmethod
        def winsorize_mad(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import winsorize_mad

            return winsorize_mad(*args, **kwargs)

        @staticmethod
        def clip_outliers(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import clip_outliers

            return clip_outliers(*args, **kwargs)

        @staticmethod
        def rank(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import rank

            return rank(*args, **kwargs)

        @staticmethod
        def label_by_quantile(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import label_by_quantile

            return label_by_quantile(*args, **kwargs)


def tracked(
    category: str,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function so calls inside ResearchRun are recorded."""

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name or func.__name__
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run = _CURRENT_RUN.get()
            if run is not None:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                run.record_step(step_name, category, _recordable_params(bound.arguments))
            return func(*args, **kwargs)

        return wrapper

    return decorate


def current_run() -> ResearchRun | None:
    """Return the current research run, if any."""
    return _CURRENT_RUN.get()


@contextmanager
def suspend_tracking():
    """Temporarily disable nested ResearchRun step recording."""

    token = _CURRENT_RUN.set(None)
    try:
        yield
    finally:
        _CURRENT_RUN.reset(token)


def _strategy_timestamp(strategy: Any) -> pd.Timestamp:
    current_datetime = getattr(strategy, "current_datetime", None)
    if callable(current_datetime):
        try:
            return pd.Timestamp(current_datetime())
        except Exception:
            pass
    data = getattr(strategy, "data", None)
    now = getattr(data, "now", None)
    if now is not None:
        try:
            return pd.Timestamp(now)
        except Exception:
            pass
    datetime_line = getattr(data, "datetime", None)
    if datetime_line is not None:
        try:
            return pd.Timestamp(datetime_line[0])
        except Exception:
            pass
    index = getattr(data, "index", None)
    if index is not None:
        try:
            position = max(0, len(data) - 1)
            return pd.Timestamp(index[position])
        except Exception:
            pass
    return pd.NaT


def _slice_multiindex_weights(
    weights: pd.Series,
    timestamp: pd.Timestamp,
    offset: int,
) -> pd.Series:
    level = weights.index.names[0] if weights.index.names[0] is not None else 0
    selected = _select_weight_timestamp(weights.index.get_level_values(0), timestamp, offset)
    if selected is None:
        return pd.Series(dtype="float64", name=weights.name or "weight")
    sliced = weights.xs(selected, level=level)
    sliced.name = weights.name or "weight"
    return sliced


def _slice_wide_weights(
    weights: pd.DataFrame,
    timestamp: pd.Timestamp,
    offset: int,
) -> pd.Series:
    selected = _select_weight_timestamp(weights.index, timestamp, offset)
    if selected is None:
        return pd.Series(dtype="float64", name="weight")
    return weights.loc[selected].dropna().rename("weight")


def _slice_long_weights(
    weights: pd.DataFrame,
    timestamp: pd.Timestamp,
    offset: int,
) -> pd.Series:
    selected = _select_weight_timestamp(pd.Index(weights["timestamp"]), timestamp, offset)
    if selected is None:
        return pd.Series(dtype="float64", name="weight")
    mask = _timestamp_match_mask(pd.Index(weights["timestamp"]), selected)
    frame = weights.loc[mask]
    if frame.empty:
        return pd.Series(dtype="float64", name="weight")
    return pd.Series(
        frame["weight"].to_numpy(dtype="float64"),
        index=frame["symbol"].astype(str),
        name="weight",
    )


def _select_weight_timestamp(
    values: pd.Index,
    timestamp: pd.Timestamp,
    offset: int,
) -> Any | None:
    if pd.isna(timestamp):
        return None
    unique_values = pd.Index(values).drop_duplicates()
    if len(unique_values) == 0:
        return None
    date_matches = [value for value in unique_values if _same_date(value, timestamp)]
    if date_matches:
        ordered = sorted(date_matches, key=_timestamp_sort_key)
        idx = len(ordered) - 1 + offset
        return ordered[idx] if 0 <= idx < len(ordered) else None
    eligible = [
        value
        for value in unique_values
        if _timestamp_sort_key(value) <= _timestamp_sort_key(timestamp)
    ]
    if not eligible:
        return None
    ordered = sorted(eligible, key=_timestamp_sort_key)
    idx = len(ordered) - 1 + offset
    return ordered[idx] if 0 <= idx < len(ordered) else None


def _same_date(value: Any, timestamp: pd.Timestamp) -> bool:
    try:
        return pd.Timestamp(value).date() == pd.Timestamp(timestamp).date()
    except Exception:
        return False


def _timestamp_sort_key(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _timestamp_match_mask(values: pd.Index, selected: Any) -> list[bool]:
    return [_timestamp_sort_key(value) == _timestamp_sort_key(selected) for value in values]


def _recordable_params(arguments: dict[str, Any]) -> dict[str, Any]:
    skipped = {
        "data",
        "features",
        "frame",
        "scores",
        "selected",
        "weights",
        "exposures",
        "returns",
    }
    return {
        name: _json_safe(value)
        for name, value in arguments.items()
        if name not in skipped and _is_recordable(value)
    }


def _is_recordable(value: Any) -> bool:
    if isinstance(value, pd.DataFrame | pd.Series):
        return False
    if callable(value):
        return False
    return True


def _flatten_step_params(steps: Sequence[ResearchStep]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.name] = counts.get(step.name, 0) + 1
        prefix = step.name if counts[step.name] == 1 else f"{step.name}_{counts[step.name]}"
        for key, value in step.params.items():
            params[f"{prefix}.{key}"] = value
    return params


def _series_dict(values: Any) -> dict[str, Any]:
    series = pd.Series(values)
    return {str(key): _json_safe(value) for key, value in series.to_dict().items()}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


__all__ = [
    "ResearchResult",
    "ResearchRun",
    "ResearchStep",
    "current_run",
    "tracked",
]
