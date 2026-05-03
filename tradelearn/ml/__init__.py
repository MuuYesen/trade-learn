"""ML public facade."""

from __future__ import annotations

from tradelearn.ml.causal import CausalSelector
from tradelearn.ml.registry import ModelLoader, ModelRegistry, model_uri

__all__ = [
    "CausalSelector",
    "AutoML",
    "ModelLoader",
    "ModelRegistry",
    "model_uri",
]


def __getattr__(name: str):
    if name == "AutoML":
        from tradelearn.ml.automl import AutoML

        return AutoML
    raise AttributeError(f"module 'tradelearn.ml' has no attribute {name!r}")
