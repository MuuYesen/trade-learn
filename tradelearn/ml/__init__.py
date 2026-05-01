"""ML public facade."""

from __future__ import annotations

import importlib
from typing import Any

from tradelearn.ml.causal import CausalSelector
from tradelearn.ml.features import FeatureStore, feature
from tradelearn.ml.registry import ModelRegistry, model_uri
from tradelearn.ml.strategy import MLStrategy

__all__ = [
    "CausalSelector",
    "FeatureStore",
    "MLStrategy",
    "ModelRegistry",
    "automl",
    "feature",
    "model_uri",
]


def __getattr__(name: str) -> Any:
    if name == "automl":
        return importlib.import_module("tradelearn.ml.automl")
    raise AttributeError(f"module 'tradelearn.ml' has no attribute {name!r}")
