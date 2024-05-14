from tradelearn.query.query import Query

from tradelearn.strategy.examine.examine import Examine

from tradelearn.strategy.backtest.single import LongBacktest
from tradelearn.strategy.evaluate.evaluate import Evaluate

from tradelearn.strategy.preprocess.explore.explore import Explore
from tradelearn.trader.utils.align import Align

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import tradelearn.trader as bt


if __name__ == '__main__':

    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    baseline = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')

    rawdata = Query.history_ohlc(symbol='600520', start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
    rawdata['label'] = rawdata['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)
    fea_list = rawdata.columns.drop(['label', 'code', 'date']).tolist()

    Explore.analysis_report(rawdata)

    rawdata = Align.transform(rawdata, baseline)

    class RSI(bt.Indicator):

        lines = ("model_indi",)

        def __init__(self, stockid, fina_data, bt_begin_date, bt_end_date, fea_list):

            indi = Query.tec_indicator(fina_data, ['RSI'])

            def signal(x):
                if x < 20:
                    return True
                if x > 40:
                    return False
                return np.NAN
            indi = indi.set_index('date').map(signal)

            bt_indi = indi.query(f"date >= '{bt_begin_date}' and date < '{bt_end_date}'").values.reshape(-1)
            tmp_list = [np.NaN if fina_data['is_fake'].iloc[i] else bt_indi[i] for i in range(len(bt_indi))]

            self.lines.model_indi.array.extend(tmp_list)

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    res = LongBacktest.run(test_data=rawdata,
                           base_line=baseline,
                           model_class=RSI,
                           feature_list=fea_list,
                           begin_date=bt_begin_date,
                           end_date=bt_end_date)

    Evaluate.analysis_report(res, baseline, engine='pyfolio')



