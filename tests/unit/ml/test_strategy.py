from __future__ import annotations

import tradelearn.engine as bt
import tradelearn.ml as ml


def test_engine_facade_does_not_export_mlstrategy() -> None:
    assert "MLStrategy" not in bt.__all__
    assert not hasattr(bt, "MLStrategy")


def test_ml_facade_does_not_export_mlstrategy() -> None:
    assert "MLStrategy" not in ml.__all__
    assert not hasattr(ml, "MLStrategy")


def test_ml_facade_exports_automl_next_to_causal_selector() -> None:
    """AutoML is exposed directly from the ML facade."""

    assert ml.AutoML.__name__ == "AutoML"
    assert ml.CausalSelector.__name__ == "CausalSelector"


def test_ml_facade_does_not_export_feature_store() -> None:
    """FeatureStore was removed from the ML public surface."""

    assert "FeatureStore" not in ml.__all__
    assert "feature" not in ml.__all__
    assert not hasattr(ml, "FeatureStore")
    assert not hasattr(ml, "feature")
