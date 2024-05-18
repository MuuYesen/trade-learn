import unittest
from tradelearn.query import Query
from tradelearn.strategy.examine import Examine


class TestExplore(unittest.TestCase):

    def test_factor_compare(self):
        data = Query.read_csv('./data/000300SH_POST.csv', begin='2020-01-01', end='2023-06-21')
        res = Examine.factor_compare(data)
        print(res)

    def test_single_factor(self):
        data = Query.read_csv('./data/000300SH_POST.csv', begin='2020-01-01', end='2023-06-21')
        Examine.single_factor(data, 'alpha001_101', filename='res/examine.html')


