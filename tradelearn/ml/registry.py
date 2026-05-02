"""MLflow Model Registry helpers for MLStrategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol for user-defined model registries used by MLStrategy."""

    def load(self, reference: str) -> Any:
        """Load a model object from a user-facing reference."""
        ...


def model_uri(reference: str) -> str:
    """Return a MLflow models:/ URI from a user-facing model reference."""
    if reference.startswith("models:/"):
        return reference
    if "@" in reference:
        name, alias = reference.split("@", 1)
        if not name or not alias:
            raise ValueError("model alias reference must be 'name@alias'.")
        return f"models:/{name}@{alias}"
    if ":" in reference:
        name, stage = reference.split(":", 1)
        if not name or not stage:
            raise ValueError("model stage reference must be 'name:stage'.")
        return f"models:/{name}/{stage}"
    return f"models:/{reference}/Production"


@dataclass
class ModelRegistry:
    """Small MLflow Model Registry facade used by MLStrategy."""

    tracking_uri: str | None = None
    mlflow_module: Any | None = None

    def load(self, reference: str) -> Any:
        """Load a pyfunc model from MLflow Model Registry."""
        mlflow = self.mlflow_module or _import_mlflow()
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow.pyfunc.load_model(model_uri(reference))


def _import_mlflow() -> Any:
    import mlflow

    return mlflow


__all__ = ["ModelLoader", "ModelRegistry", "model_uri"]
