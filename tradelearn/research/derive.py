from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd


class _ProgramHistory(list):
    """Snapshot each generated population before gplearn prunes ancestors."""

    def __init__(self):
        super().__init__()
        self.candidates = []

    def append(self, population):
        generation = len(self)
        self.candidates.extend(
            {
                "generation": generation,
                "index": index,
                "expression": str(program),
                "length": int(program.length_),
                "fitness": float(program.raw_fitness_),
            }
            for index, program in enumerate(population)
        )
        super().append(population)


class SymbolicFeatureGenerator:
    """Generate symbolic features through optional gplearn.

    This class owns the heavy symbolic-transformer configuration and follows
    the sklearn-style ``fit`` / ``transform`` split to avoid train/test leakage.
    """

    def __init__(
        self,
        *,
        n_features: int = 20,
        random_state: int | None = None,
        generations: int = 3,
        population_size: int = 100,
        metric: str = "pearson",
        **kwargs: Any,
    ) -> None:
        self.n_features = int(n_features)
        self.random_state = random_state
        self.generations = int(generations)
        self.population_size = int(population_size)
        self.metric = str(metric)
        self.kwargs = dict(kwargs)
        self.transformer_: Any | None = None
        self.input_columns_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.fill_values_: pd.Series | None = None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str],
        target: str,
    ) -> SymbolicFeatureGenerator:
        """Fit the symbolic transformer on training data."""
        transformer_type = _load_symbolic_transformer()
        frame = pd.DataFrame(data)
        feature_columns = [str(column) for column in columns]
        target_name = str(target)
        clean = frame[feature_columns + [target_name]].dropna()
        if clean.empty:
            raise ValueError("symbolic feature training data is empty after dropping NA")

        transformer = transformer_type(
            n_components=self.n_features,
            generations=self.generations,
            population_size=self.population_size,
            random_state=self.random_state,
            metric=self.metric,
            feature_names=feature_columns,
            **self.kwargs,
        )
        # gplearn 0.4.2 (the Python 3.10 release) calls the estimator validation
        # method removed by newer sklearn. Adapt only this owned instance, with
        # sklearn's public validator; no class/global patch or formula change.
        if type(transformer).__module__.startswith("gplearn.") and not hasattr(
            transformer, "_validate_data"
        ):
            import gplearn

            if gplearn.__version__ == "0.4.2":
                transformer._validate_data = partial(_validate_symbolic_data, transformer)
        # A fresh transformer is fitted each time. Capture populations at the
        # append boundary before upstream's memory pruning. No global patch or
        # parallel worker instrumentation; qualified against gplearn 0.4.3.
        if hasattr(transformer, "warm_start"):
            transformer.warm_start = True
            transformer._programs = _ProgramHistory()
            transformer.run_details_ = {
                key: []
                for key in (
                    "generation",
                    "average_length",
                    "average_fitness",
                    "best_length",
                    "best_fitness",
                    "best_oob_fitness",
                    "generation_time",
                )
            }
        transformer.fit(clean[feature_columns], clean[target_name])
        self.transformer_ = transformer
        self.input_columns_ = feature_columns
        self.feature_names_ = [f"symbolic_{i + 1}" for i in range(self.n_features)]
        self.fill_values_ = clean[feature_columns].median()
        self.input_dtypes_ = {column: str(clean[column].dtype) for column in feature_columns}
        self.training_ = {
            "rows": len(clean),
            "dropped_rows": len(frame) - len(clean),
            "start": str(clean.index.min()),
            "end": str(clean.index.max()),
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the fitted symbolic transformer."""
        if self.transformer_ is None or self.input_columns_ is None:
            raise RuntimeError("fit() must be called before transform()")
        frame = pd.DataFrame(data)
        feature_frame = frame[self.input_columns_]
        fill_values = self.fill_values_
        if fill_values is not None:
            feature_frame = feature_frame.fillna(fill_values)
        values = np.asarray(self.transformer_.transform(feature_frame))
        columns_out = self.feature_names_ or [f"symbolic_{i + 1}" for i in range(values.shape[1])]
        return pd.DataFrame(values, index=frame.index, columns=columns_out)

    def fit_transform(
        self,
        data: pd.DataFrame,
        *,
        columns: Sequence[str],
        target: str,
    ) -> pd.DataFrame:
        """Fit the symbolic transformer and return generated features."""
        return self.fit(data, columns=columns, target=target).transform(data)

    def to_dict(self) -> dict[str, Any]:
        """Describe fitted expressions and retained search evidence.

        Exact inference restore requires the fitted object and gplearn; these
        strings are inspectable evidence, not a dependency-free evaluator.
        gplearn prunes ancestral programs; missing ancestors stay explicit.
        """
        if self.transformer_ is None:
            raise RuntimeError("fit() must be called before to_dict()")
        generations = getattr(self.transformer_, "_programs", [])
        candidates = []
        for generation, programs in enumerate(generations):
            if programs is None:
                continue
            for index, program in enumerate(programs):
                if program is not None:
                    candidates.append(
                        {
                            "generation": generation,
                            "index": index,
                            "expression": str(program),
                            "length": int(program.length_),
                            "fitness": float(program.raw_fitness_),
                        }
                    )
        captured = getattr(generations, "candidates", None)
        if captured is not None:
            candidates = captured
        selected = getattr(self.transformer_, "_best_programs", [])
        return {
            "params": self.get_params(),
            "input_columns": self.input_columns_,
            "input_dtypes": self.input_dtypes_,
            "training": self.training_,
            "fill_values": self.fill_values_.to_dict(),
            "expressions": [str(program) for program in selected],
            "candidates": candidates,
            "candidate_capture": "complete" if captured is not None else "retained_only",
            "pruned_candidates": 0
            if captured is not None
            else sum(
                sum(p is None for p in programs) for programs in generations if programs is not None
            ),
            "runtime_dependency": "gplearn",
            "feature_names": self.feature_names_,
        }

    def get_params(self) -> dict[str, Any]:
        """Return serializable generator parameters for tracking artifacts."""
        params = {
            "type": type(self).__name__,
            "n_features": self.n_features,
            "random_state": self.random_state,
            "generations": self.generations,
            "population_size": self.population_size,
            "metric": self.metric,
        }
        params.update(self.kwargs)
        return params



def _validate_symbolic_data(estimator, X, y="no_validation", reset=True, **options):
    """Use sklearn's public validator for an owned legacy gplearn instance.

    A module-level partial remains pickleable with the fitted estimator. It
    retains sklearn feature/count/finite checks rather than bypassing validation.
    """
    from sklearn.utils.validation import validate_data

    return validate_data(estimator, X, y, reset=reset, **options)


def _load_symbolic_transformer():
    try:
        from gplearn.genetic import SymbolicTransformer
    except ImportError as exc:
        raise ImportError(
            "SymbolicFeatureGenerator requires optional dependency `gplearn`. "
            "Install it with `pip install gplearn`."
        ) from exc
    return SymbolicTransformer


__all__ = ["SymbolicFeatureGenerator"]
