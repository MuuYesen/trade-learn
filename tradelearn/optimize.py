"""Optional Optuna-backed parameter search helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


class OptunaSearch:
    """Thin optional wrapper around an Optuna study."""

    def __init__(
        self,
        objective: Callable[[Any], float],
        *,
        n_trials: int = 100,
        direction: str = "maximize",
        study_name: str | None = None,
        storage: str | None = None,
        load_if_exists: bool = False,
        sampler: Any | None = None,
        pruner: Any | None = None,
        **study_kwargs: Any,
    ) -> None:
        self.objective = objective
        self.n_trials = int(n_trials)
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.load_if_exists = bool(load_if_exists)
        self.sampler = sampler
        self.pruner = pruner
        self.study_kwargs = dict(study_kwargs)

    def run(self, **optimize_kwargs: Any):
        """Create and optimize an Optuna study, then return it."""
        optuna = _import_optuna()
        kwargs = {
            "direction": self.direction,
            "study_name": self.study_name,
            "storage": self.storage,
            "load_if_exists": self.load_if_exists,
            **self.study_kwargs,
        }
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler
        if self.pruner is not None:
            kwargs["pruner"] = self.pruner
        study = optuna.create_study(**kwargs)
        study.optimize(self.objective, n_trials=self.n_trials, **optimize_kwargs)
        return study


class OptunaBacktestSearch:
    """Optuna search wrapper for Engine/Lite backtest factories."""

    def __init__(
        self,
        *,
        backtest_factory: Callable[[dict[str, Any]], Any],
        param_space: Mapping[str, Any],
        metric: str | Callable[[Any], float],
        n_trials: int = 100,
        direction: str = "maximize",
        **study_kwargs: Any,
    ) -> None:
        self.backtest_factory = backtest_factory
        self.param_space = dict(param_space)
        self.metric = metric
        self.search = OptunaSearch(
            self._objective,
            n_trials=n_trials,
            direction=direction,
            **study_kwargs,
        )

    def run(self, **optimize_kwargs: Any):
        """Run parameter search and return the Optuna study."""
        return self.search.run(**optimize_kwargs)

    def _objective(self, trial: Any) -> float:
        params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in self.param_space.items()
        }
        backtest = self.backtest_factory(params)
        result = backtest.run() if hasattr(backtest, "run") else backtest(params)
        return _metric_value(result, self.metric)


def _import_optuna():
    try:
        optuna = import_module("optuna")
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for this feature. Install with "
            "`pip install trade-learn[optuna]`."
        ) from exc
    if optuna is None:
        raise ImportError(
            "Optuna is required for this feature. Install with "
            "`pip install trade-learn[optuna]`."
        )
    return optuna


def _suggest_param(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, Mapping):
        kind = spec.get("type", "float")
        if kind == "int":
            return trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        if kind == "float":
            kwargs = {
                "step": spec.get("step"),
                "log": bool(spec.get("log", False)),
            }
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec["choices"]))
        raise ValueError(f"unsupported Optuna param type {kind!r}")
    if isinstance(spec, tuple) and len(spec) in {2, 3}:
        low, high = spec[0], spec[1]
        step = spec[2] if len(spec) == 3 else None
        if isinstance(low, int) and isinstance(high, int):
            kwargs = {} if step is None else {"step": int(step)}
            return trial.suggest_int(name, low, high, **kwargs)
        kwargs = {} if step is None else {"step": float(step)}
        return trial.suggest_float(name, float(low), float(high), **kwargs)
    if isinstance(spec, list | tuple):
        return trial.suggest_categorical(name, list(spec))
    raise TypeError(f"unsupported Optuna param spec for {name!r}")


def _metric_value(result: Any, metric: str | Callable[[Any], float]) -> float:
    if callable(metric):
        return float(metric(result))
    if isinstance(result, Mapping):
        return float(result[metric])
    if hasattr(result, metric):
        return float(getattr(result, metric))
    if hasattr(result, "__getitem__"):
        return float(result[metric])
    raise KeyError(f"metric {metric!r} not found in backtest result")


__all__ = ["OptunaBacktestSearch", "OptunaSearch"]
