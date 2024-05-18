from tradelearn.query import Query
from tradelearn.strategy.preprocess.explore import Explore
from tradelearn.trader.signal import Signal
from tradelearn.strategy.backtest.single import LongBacktest
from tradelearn.strategy.evaluate import Evaluate

import numpy as np

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    baseline = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')

    rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
    rawdata['label'] = rawdata['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)

    Explore.analysis_report(rawdata)

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    class RSI(Signal):

        def __init__(self, stockid, raw_data, bt_begin_date, bt_end_date, param_dict):
            indi = Query.tec_indicator(raw_data, ['RSI'])

            def signal(x):
                if x < 20:
                    return True
                if x > 40:
                    return False
                return np.NAN
            indi = indi.set_index('date').map(signal)
            bt_indi = indi.query(f"date >= '{bt_begin_date}' and date < '{bt_end_date}'")

            self.set_signal(bt_indi)

    param_dict = {}

    res = LongBacktest.run(RSI, param_dict, rawdata, baseline, bt_begin_date, bt_end_date)
    Evaluate.analysis_report(res, baseline, engine='pyfolio')



