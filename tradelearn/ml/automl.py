"""Lightweight AutoML facade."""

import pandas as pd
from sklearn.model_selection import train_test_split


class AutoML:
    """AutoML class for automating machine learning tasks."""

    @staticmethod
    def lazy_predict(
        data: pd.DataFrame,
        *,
        target: str = "label",
        features: list[str] | tuple[str, ...] | None = None,
        classifier_cls=None,
    ):
        """Performs lazy prediction on the provided raw data.

        Args:
            data (pd.DataFrame): Dataset containing feature columns and target.

        Returns:
            models: The trained models and their predictions.
        """
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


__all__ = ["AutoML"]
