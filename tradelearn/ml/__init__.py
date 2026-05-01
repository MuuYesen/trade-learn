"""ML public facade."""

from __future__ import annotations

from tradelearn.ml.causal import CausalSelector
from tradelearn.ml.features import FeatureStore, feature
from tradelearn.ml.pipeline import (
    FactorTransformer,
    ModelAdapter,
    PipelineResult,
    StrategyPipeline,
)
from tradelearn.ml.registry import ModelRegistry, model_uri
from tradelearn.ml.strategy import MLStrategy
from tradelearn.portfolio import EqualWeightOptimizer, RiskPolicy, TopKSelector

__all__ = [
    "CausalSelector",
    "EqualWeightOptimizer",
    "AutoML",
    "FactorTransformer",
    "FeatureStore",
    "MLStrategy",
    "ModelAdapter",
    "ModelRegistry",
    "PipelineResult",
    "RiskPolicy",
    "StrategyPipeline",
    "TopKSelector",
    "feature",
    "model_uri",
]


def __getattr__(name: str):
    if name == "AutoML":
        from tradelearn.ml.automl import AutoML

        return AutoML
    raise AttributeError(f"module 'tradelearn.ml' has no attribute {name!r}")
