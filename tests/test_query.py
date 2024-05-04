import unittest
import numpy as np
import pandas as pd

from traderpy.query.query import Query


class TestQuery(unittest.TestCase):

    def test_read_csv(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        print(data)

    def test_query_incators(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        Query.tec_indicator(data)
