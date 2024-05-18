import unittest
from tradelearn.query import Query
from tradelearn.automl import AutoML


class TestExplore(unittest.TestCase):

    def test_lazy_predict(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        data['label'] = data['close'].pct_change(-1).apply(lambda x: 1 if x>0 else -1)

        model = AutoML.lazy_predict(data)
        print(model)


