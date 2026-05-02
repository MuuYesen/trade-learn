from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from tradelearn.research.run import current_run


@runtime_checkable
class Transformer(Protocol):
    """Protocol for user-defined research preprocessing steps."""

    def fit(self, data: Any) -> Any:
        """Fit state from training data."""
        ...

    def transform(self, data: Any) -> Any:
        """Transform data using fitted state."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return serializable parameters for tracking."""
        ...


class Pipeline:
    """Sequential train/test-safe research transformer pipeline."""

    def __init__(self, steps: Sequence[Any]) -> None:
        self.steps = [_normalise_step(step) for step in steps]

    def fit(self, data: Any) -> Pipeline:
        """Fit each step sequentially using training data."""
        _record_pipeline_step(self)
        current = data
        for _name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Pipeline step {type(step).__name__} must provide transform()")
            if hasattr(step, "fit"):
                step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, data: Any) -> Any:
        """Transform data using the fitted pipeline state."""
        current = data
        for _name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Pipeline step {type(step).__name__} must provide transform()")
            current = step.transform(current)
        return current

    def fit_transform(self, data: Any) -> Any:
        """Fit each step and return transformed training data."""
        _record_pipeline_step(self)
        current = data
        for _name, step in self.steps:
            if hasattr(step, "fit_transform"):
                current = step.fit_transform(current)
                continue
            if not hasattr(step, "transform"):
                raise TypeError(f"Pipeline step {type(step).__name__} must provide transform()")
            if hasattr(step, "fit"):
                step.fit(current)
            current = step.transform(current)
        return current

    def get_params(self) -> dict[str, Any]:
        """Return serializable pipeline parameters for tracking artifacts."""
        return {
            "steps": [
                step.get_params() if hasattr(step, "get_params") else {"type": type(step).__name__}
                for _name, step in self.steps
            ]
        }


def _normalise_step(step: Any) -> tuple[str, Any]:
    if isinstance(step, tuple):
        if len(step) != 2:
            raise ValueError("Pipeline named steps must be (name, step) tuples")
        name, transformer = step
        return str(name), transformer
    return type(step).__name__, step


def _record_pipeline_step(pipeline: Pipeline) -> None:
    run = current_run()
    if run is not None:
        run.record_step(
            "Pipeline",
            "preprocess",
            {"steps": [name for name, _step in pipeline.steps]},
        )


__all__ = ["Pipeline", "Transformer"]
