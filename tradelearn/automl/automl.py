import pandas as pd

from tradelearn.automl.common.lazy_predict import LazyClassifier
from sklearn.model_selection import train_test_split


class AutoML:
    """AutoML class for automating machine learning tasks."""

    def __init__(self):
        """Initializes the AutoML class."""
        pass  # Placeholder for future initialization code

    @staticmethod
    def lazy_predict(rawdata: pd.DataFrame):
        """Performs lazy prediction on the provided raw data.

        Args:
            rawdata (pd.DataFrame): The input data containing features and labels.

        Returns:
            models: The trained models and their predictions.
        """
        dataY = rawdata['label']  # Target variable for prediction
        dataX = rawdata.drop(['label'], axis=1)  # Features for the model

        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=.5, random_state=123)

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        return models  # Returns the trained models and their predictions
