from __future__ import annotations

import pandas as pd

from tradelearn.ml import AutoML


def test_automl_lazy_predict_splits_dataset_by_target_and_features() -> None:
    """AutoML accepts one dataset plus target/features configuration."""

    seen: dict[str, object] = {}

    class FakeLazyClassifier:
        def __init__(self, **kwargs) -> None:
            seen["kwargs"] = kwargs

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

    models = AutoML.lazy_predict(
        data,
        target="label",
        features=["feature_b"],
        classifier_cls=FakeLazyClassifier,
    )

    assert seen["x_columns"] == ["feature_b"]
    assert seen["y_name"] == "label"
    assert models.index.tolist() == ["FakeModel"]
