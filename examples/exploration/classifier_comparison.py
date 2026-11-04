"""Exploratory classifier leaderboard; random split is NOT temporal validation.

This example intentionally retains the former AutoML convenience behavior.
For financial evaluation use explicit chronological splits, label maturity and
train-only fitting through the existing research/selector/model interfaces.
The return value is a leaderboard, not a trained model or trading signal.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def compare_classifiers(
    data: pd.DataFrame, *, target: str = "label", features=None, classifier_cls=None
):
    """Compare classifiers on a random half-split, for exploration only."""
    frame = pd.DataFrame(data)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not found")
    feature_names = (
        [str(column) for column in features]
        if features is not None
        else [str(column) for column in frame.columns if str(column) != target]
    )
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"feature column(s) not found: {missing}")
    data_y = frame[target]
    data_x = frame.loc[:, feature_names]

    x_train, x_test, y_train, y_test = train_test_split(
        data_x,
        data_y,
        test_size=0.5,
        random_state=123,
    )

    if classifier_cls is None:
        from tradelearn.ml._lazy_predict import LazyClassifier

        classifier_cls = LazyClassifier

    clf = classifier_cls(verbose=0, ignore_warnings=True, custom_metric=None)
    models, _predictions = clf.fit(x_train, x_test, y_train, y_test)

    return models
