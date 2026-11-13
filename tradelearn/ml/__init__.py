"""ML public facade."""

from __future__ import annotations

from tradelearn.ml.artifacts import load_bundle, predict_bundle, save_bundle
from tradelearn.ml.causal import CausalSelector
from tradelearn.ml.registry import ModelLoader, ModelRegistry, model_uri

__all__ = [
    "CausalSelector",
    "ModelLoader",
    "ModelRegistry",
    "model_uri",
]

__all__ += ["save_bundle", "load_bundle", "predict_bundle"]
