import unittest
from tradelearn.query.query import Query


class TestQuery(unittest.TestCase):

    def test_read_csv(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        print(data)

    def test_query_alpha101(self):
        tn_begin_date = '2017-01-01'
        tn_end_date = '2022-06-22'

        rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq',
                                     engine='tdx')
        res = Query.alphas101(rawdata, ['alpha001'])
        print(res)

    def test_query_incators(self):
        tn_begin_date = '2017-01-01'
        tn_end_date = '2022-06-22'

        rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq',
                                     engine='tdx')
        res = Query.tec_indicator(rawdata, ['ATR', 'RSI'])
        print(res)
