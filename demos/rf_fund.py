from tradelearn.query import Query
from tradelearn.strategy.backtest import Backtest, Strategy

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':

    tn_begin_date = '2017-01-01'
    tn_end_date = '2022-06-22'

    rawdata = None
    for i in range(7):
        temp = Query.history_ohlc(symbol='60052' + str(i), start=tn_begin_date, end=tn_end_date, adjust='hfq', engine='tdx')
        if temp is None:
            continue

        temp['label'] = temp['close'].pct_change(periods=1).shift(-1).map(lambda x: 1 if x > 0 else -1)
        rawdata = pd.concat([rawdata, temp], axis=0)

    bt_begin_date = '2020-01-01'
    bt_end_date = '2022-06-22'

    class RandomForest(Strategy):

        def init(self):
            data = self.data.df.swaplevel(0, 1, axis=1).stack().reset_index(level=1)
            fea_list = data.columns.drop(['label', 'code']).tolist()

            train_data = data.query(f"date >= '{tn_begin_date}' and date < '{bt_begin_date}'")
            bt_x_train, bt_y_train = train_data[fea_list], train_data['label']
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            model.fit(bt_x_train, bt_y_train)

            test_data = data.query(f"date >= '{bt_begin_date}' and date < '{bt_end_date}'")
            ind_df = pd.DataFrame({'date': data.index.unique()}).set_index('date')
            for symbol in test_data['code'].unique():
                bt_x_test = test_data.query(f"code == '{symbol}'")[fea_list]
                pre_proba = model.predict_proba(bt_x_test)[:, 1]
                ind_df = pd.merge(pd.DataFrame(pre_proba, index=bt_x_test.index, columns=[symbol]),
                                  ind_df, on=['date'], how='right')

            self.proba = self.I(ind_df, overlay=False)

        def next(self):
            self.alloc.assume_zero()

            proba = self.proba.df.iloc[-1]
            bucket = self.alloc.bucket['equity']

            bucket.append(proba.sort_values(ascending=False))
            bucket.trim(limit=3)
            bucket.weight_explicitly(weight=1/3)
            bucket.apply(method='update')

            self.rebalance(cash_reserve=0.1)


    btdata = rawdata.pivot_table(index='date', columns='code').swaplevel(0, 1, axis=1)
    btdata = btdata.sort_values(by='code', axis=1).fillna(method='ffill')

    bt = Backtest(btdata, RandomForest, cash=1000000, commission=.002, trade_on_close=False)
    bt.run()
    bt.plot(plot_volume=True, superimpose=False, plot_allocation=True)



