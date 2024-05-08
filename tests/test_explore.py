import unittest
import numpy as np
import pandas as pd

from tradelearn.query.query import Query
from tradelearn.strategy.preprocess.explore.explore import Explore


class TestExplore(unittest.TestCase):

    def test_explore_report(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')



