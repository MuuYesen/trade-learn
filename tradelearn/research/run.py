from __future__ import annotations

import contextvars
import inspect
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
            scores=None if scores is None else pd.Series(scores, name=getattr(scores, "name", "score")),
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
        def winsorize(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import winsorize

            return winsorize(*args, **kwargs)

        @staticmethod
        def neutralize(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import neutralize

            return neutralize(*args, **kwargs)

        @staticmethod
        def standardize(*args: Any, **kwargs: Any) -> Any:
            from tradelearn.research.preprocess import standardize

            return standardize(*args, **kwargs)


def tracked(category: str, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
        "target",
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
