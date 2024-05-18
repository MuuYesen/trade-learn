import unittest

from tradelearn.query import Query
from tradelearn.strategy.preprocess.derive import Derive

class TestDerive(unittest.TestCase):

    def test_derive(self):
        tn_begin_date = '2017-01-01'
        tn_end_date = '2022-06-22'

        data = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq',
                                  engine='tdx')
        res = Derive.generic_generate(data, random=21)
        print(res)


