import pandas as pd

from tradelearn.automl.common.lazy_predict import LazyClassifier

from sklearn.model_selection import train_test_split


class AutoML:

    def __int__(self):
        pass

    @staticmethod
    def lazy_predict(rawdata: pd.DataFrame):

        dataY = rawdata['label']
        dataX = rawdata.drop(['label'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=.5, random_state=123)

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        return models
