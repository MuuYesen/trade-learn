from __future__ import annotations

import pandas as pd

from examples.exploration.classifier_comparison import compare_classifiers


def test_exploratory_comparison_keeps_explicit_feature_selection() -> None:
    """Exploration only: the old leaderboard remains outside the public ML API."""

    seen: dict[str, object] = {}

    class FakeLazyClassifier:
        def __init__(self, **kwargs) -> None:
            seen["kwargs"] = kwargs
            self.failures_ = {"BadModel": {"stage": "fit"}}
            self.score_errors_ = {}

        def fit(self, x_train, x_test, y_train, y_test):
            seen["x_columns"] = list(x_train.columns)
            seen["y_name"] = y_train.name
            return pd.DataFrame({"Accuracy": [1.0]}, index=["FakeModel"]), pd.DataFrame()

    data = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
            "ignored": ["x", "y", "z", "w"],
            "label": [0, 1, 1, 0],
        }
    )

    models = compare_classifiers(
        data,
        target="label",
        features=["feature_b"],
        classifier_cls=FakeLazyClassifier,
    )

    assert seen["x_columns"] == ["feature_b"]
    assert seen["y_name"] == "label"
    assert models.index.tolist() == ["FakeModel"]
    assert models.attrs["failures"]["BadModel"]["stage"] == "fit"
    assert models.attrs["score_errors"] == {}


def test_exploratory_comparison_rejects_unknown_target_and_features():
    import pytest

    data = pd.DataFrame({"label": [0, 1], "feature": [1.0, 2.0]})
    with pytest.raises(ValueError, match="target column"):
        compare_classifiers(data, target="absent")
    with pytest.raises(ValueError, match="feature column"):
        compare_classifiers(data, features=["absent"])
