import os
import time
import traceback
import pandas as pd
from multiprocessing import Pool

import yfinance as yf
from mootdx.quotes import Quotes

from tradelearn.query.common.alphas101 import Alphas101
from tradelearn.query.common.alphas191 import Alphas191
from tradelearn.query.common.tdx30 import Tdx30


class Query:

    def __init__(self):
        pass

    @staticmethod
    def read_csv(data_path, begin, end):
        data = pd.read_csv(data_path, parse_dates=['date'], dtype={'code': str}, low_memory=True, encoding='utf_8_sig')
        data = data.query(f"date >= '{begin}' and date <= '{end}'")
        return data

    @staticmethod
    def history_ohlc(symbol=None, start=None, end=None, adjust='qfq', engine='tdx'):
        if engine == 'yahoo':
            try:
                tickler = yf.Ticker(symbol)
                if adjust == 'qfq':
                    auto_adjust = True
                data = tickler.history(start=start, end=end, interval="1d", auto_adjust=auto_adjust)
                data = data.reset_index()
                data.columns = [fname.lower() for fname in data.columns]
                data['code'] = symbol
                data['date'] = data['date'].dt.tz_localize(None)
            except:
                data = None

        if engine == 'tdx':
            try:
                client = Quotes.factory(market='std', multithread=True, heartbeat=True, timeout=15, auto_retry=True)
                data = client.ohlc(symbol=symbol, begin=start, end=end, adjust=adjust)
                data = data.drop(['date'], axis=1).reset_index()
                data[['open', 'close', 'high', 'low']] = data[['open', 'close', 'high', 'low']].apply(lambda x: x / data['factor'], axis=0)
                if not data is None:
                    data['vwap'] = data.amount / data.volume / 100
            except:
                data = None

        return data

    @staticmethod
    def _calc_func(func, m, kwargs={}):
        t1 = time.time()
        alpha = func(**kwargs)
        t2 = time.time()
        print(f"{m} time {t2 - t1}")
        return alpha

    # @staticmethod
    # def _calc_func(func, m):
    #     t1 = time.time()
    #     alpha = func()
    #     t2 = time.time()
    #     print(f"{m} time {t2 - t1}")
    #     return alpha

    @staticmethod
    def tec_indicator(stock_data: pd.DataFrame, alpha_name: list = None, **kwargs):
        mt = Tdx30(stock_data)

        if alpha_name:
            methods = alpha_name
        else:
            methods = (list(filter(lambda m: (not m.startswith("_")) and callable(getattr(mt, m)),
                            dir(mt))))
        res_list = []
        pool = Pool(processes=os.cpu_count())
        for m in methods:
            fac_func = getattr(mt, m)
            try:
                def print_error(value):
                    print("error: ", value)
                res = pool.apply_async(Query._calc_func, (fac_func, m, kwargs), error_callback=print_error)
                res_list.append([m, res])
            except:
                traceback.print_exc()

        res = pd.DataFrame({'date':[]})
        for m, r in res_list:
            ind_nb = r.get()
            df = pd.DataFrame({'date': stock_data['date'], m: ind_nb})
            res = pd.merge(res, df, how='outer', on=['date'])

        pool.close()
        pool.join()

        return res

    @staticmethod
    def alphas101(stock_data: pd.DataFrame, alpha_name: list = None):

        stock_data = stock_data.pivot(index='date', columns='code')

        af = Alphas101(stock_data)

        if alpha_name:
            methods = alpha_name
        else:
            methods = (list(filter(lambda m: m.startswith("alpha") and callable(getattr(af, m)),
                            dir(af))))

        res_list = []
        pool = Pool(processes=os.cpu_count())
        for m in methods:
            fac_func = getattr(af, m)
            try:
                def print_error(value):
                    print("error: ", value)
                res = pool.apply_async(Query._calc_func, (fac_func, m), error_callback=print_error)
                res_list.append([m, res])
            except:
                traceback.print_exc()

        res = pd.DataFrame({'date':[], 'code':[]})
        for m, r in res_list:
            df = r.get()
            df['date'] = df.index
            df = df.melt(id_vars='date', value_vars=df.columns.drop('date'), value_name=m)
            df.rename(columns={m: m + '_101'}, inplace=True)
            res = pd.merge(res, df, how='outer', on=['date', 'code'])

        pool.close()
        pool.join()

        return res

    @staticmethod
    def alphas191(stock_data: pd.DataFrame, bench_data: pd.DataFrame, alpha_name: list = None):
        stock_data = stock_data.pivot(index='date', columns='code')

        af = Alphas191(stock_data, bench_data)

        if alpha_name:
            methods = alpha_name
        else:
            methods = (list(filter(lambda m: m.startswith("alpha") and callable(getattr(af, m)),
                            dir(af))))

        res_list = []
        pool = Pool(processes=os.cpu_count())
        for m in methods:
            print(m)
            fac_func = getattr(af, m)
            try:
                def print_error(value):
                    print("error: ", value)
                res = pool.apply_async(Query._calc_func, (fac_func, m), error_callback=print_error)
                res_list.append([m, res])
            except:
                traceback.print_exc()

        res = pd.DataFrame({'date':[], 'code':[]})
        for m, r in res_list:
            df = r.get()
            print(m, type(df), df.columns)
            df['date'] = df.index
            df = df.melt(id_vars='date', value_vars=df.columns.drop('date'), value_name=m)
            df.rename(columns={m: m + '_191'}, inplace=True)
            res = pd.merge(res, df, how='outer', on=['date', 'code'])

        pool.close()
        pool.join()

        return res