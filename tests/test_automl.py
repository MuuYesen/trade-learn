import unittest

from tradelearn.query.query import Query
from tradelearn.automl.lazy_predict import LazyClassifier

from sklearn.model_selection import train_test_split


class TestExplore(unittest.TestCase):

    def test_lazy_classifer(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        y = data['close'].pct_change(-1).apply(lambda x: 1 if x>0 else -1)
        X = data.drop(['close'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=123)

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        print(models)


