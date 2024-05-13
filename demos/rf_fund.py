from tradelearn.query.query import Query

from tradelearn.strategy.backtest.fund import LongBacktest

from tradelearn.strategy.evaluate.evaluate import Evaluate

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings('ignore')

import tradelearn.trader as bt
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':

    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    baseline = Query.history_ohlc(symbol='000001.SS', start=tn_begin_date, end=tn_end_date, engine='yahoo')  ## 两个接口都是右开区间，所有都是包括自定义

    rawdata = None
    for i in range(10):
        temp = Query.history_ohlc(symbol='60052' + str(i), start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
        if temp is None:
            continue

        temp['label'] = temp['close'].pct_change(periods=5).shift(-1).map(lambda x: 1 if x > 0 else -1)
        rawdata = pd.concat([rawdata, temp], axis=0)

    fea_list = rawdata.columns.drop(['label', 'code', 'date']).tolist()

    from tradelearn.strategy.preprocess.explore.explore import Explore
    Explore.analysis_report(rawdata)

    from tradelearn.trader.utils.align import Align
    rawdata = Align.transform(rawdata, baseline)

    class RandomForestIndicator(bt.Indicator):

        model_dict = {}

        lines = ("model_indi",)

        def __init__(self, stockid, fina_data, bt_begin_date, bt_end_date, fea_list):

            if not RandomForestIndicator.model_dict:
                train_data = fina_data.query("is_fake == False")

                for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                    bt_train_data = train_data.query(f"date >= '{date - relativedelta(months=12 * 3)}' and date < '{date}'")
                    bt_x_train, bt_y_train = bt_train_data[fea_list], bt_train_data['label']

                    model = RandomForestClassifier(random_state=42, n_jobs=-1)
                    model.fit(bt_x_train, bt_y_train)
                    RandomForestIndicator.model_dict[date.year] = model

            indi_list = []
            for date in pd.date_range(start=bt_begin_date, end=bt_end_date, freq='12MS'):
                pos_data = fina_data.query(f"code == '{stockid}' and date >= '{date}' and date < '{date + relativedelta(months=12 * 1)}'")
                bt_x_test = pos_data.set_index(['date'])[fea_list]
                pre_proba = RandomForestIndicator.model_dict[date.year].predict_proba(bt_x_test)[:, 1]

                tmp_list = [np.NaN if pos_data['is_fake'].iloc[i] else pre_proba[i] for i in range(len(pre_proba))]
                indi_list.extend(tmp_list)

            self.lines.model_indi.array.extend(indi_list)

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    res = LongBacktest.run(test_data=rawdata,
                           base_line=baseline,
                           model_class=RandomForestIndicator,
                           feature_list=fea_list,
                           begin_date=bt_begin_date,
                           end_date=bt_end_date)

    Evaluate.analysis_report(res, baseline, engine='quantstats')



