import unittest

from tradelearn.query.query import Query
from tradelearn.strategy.preprocess.derive.derive import Derive

from sklearn.model_selection import train_test_split


class TestExplore(unittest.TestCase):

    def test_lazy_classifer(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')

        y = data['close'].pct_change(-1).apply(lambda x: 1 if x>0 else -1)
        X = data.drop(['date', 'code', 'close'], axis=1)

        res = Derive.generic_generate(X, y)

        print(res)


