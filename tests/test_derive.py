import unittest

from tradelearn.query.query import Query
from tradelearn.strategy.preprocess.derive.derive import Derive

class TestExplore(unittest.TestCase):

    def test_lazy_classifer(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        res = Derive.generic_generate(data)
        print(res)


