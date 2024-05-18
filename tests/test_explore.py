import unittest
from tradelearn.query import Query
from tradelearn.strategy.preprocess.explore import Explore


class TestExplore(unittest.TestCase):

    def test_explore_report(self):
        tn_begin_date = '2017-01-01'
        tn_end_date = '2022-06-22'

        rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq',
                                     engine='tdx')
        Explore.analysis_report(rawdata, filename='res/explore.html')