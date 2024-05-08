import unittest
import numpy as np
import pandas as pd

from tradelearn.query.query import Query
from tradelearn.strategy.examine.examine import Examine


class TestExplore(unittest.TestCase):

    def test_factor_compare(self):
        data = pd.read_csv('./data/000300SH_POST.csv', index_col=0, parse_dates=['date'], dtype={'code': str}, low_memory=True, encoding='utf_8_sig')
        res = Examine.factor_compare(data)
        print(res)

    def test_single_factor(self):
        data = pd.read_csv('./data/000300SH_POST.csv', index_col=0, parse_dates=['date'], dtype={'code': str}, low_memory=True, encoding='utf_8_sig')
        Examine.single_factor(data, 'alpha001_101')


