import os
import time
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool

import yfinance as yf
from mootdx.quotes import Quotes
from tradelearn.query.tvDatafeed.main import TvDatafeed, Interval

from tradelearn.query.alpha.alphas101 import Alphas101
from tradelearn.query.alpha.alphas191 import Alphas191
from tradelearn.query.tec.mytt.tdx30 import Tdx30


class Query:
    """A class to handle data queries and calculations for stock market analysis."""

    def __init__(self):
        """Initializes the Query class."""
        pass

    @staticmethod
    def read_csv_from_tongdaxin(data_path_list: list, begin: str = None, end: str = None, data_symbol_list: list = []):
        """Reads a CSV file and filters the data by date range.

        Args:
            data_path (str): The path to the CSV file.
            begin (str): The start date for filtering (inclusive).
            end (str): The end date for filtering (inclusive).

        Returns:
            pd.DataFrame: The filtered data as a DataFrame.
        """
        path_list, symbol_list = data_path_list, data_symbol_list

        if isinstance(path_list, str):
            path_list = [path_list]
        if isinstance(symbol_list, str):
            symbol_list = [symbol_list]
  
        rename_dict = {'日期':'date', '开盘':'open', '最高':'high', '最低':'low', '收盘':'close',
                       '成交量':'volume', '持仓量':'open_interest', '结算价':'settlement_price'}		
        date_key, symbol_key = 'date', 'code'

        data_list = []
        for i, data_path in enumerate(path_list):

            data = pd.read_csv(data_path, low_memory=True, encoding='gbk', skiprows=2, skipfooter=1, names=rename_dict.keys())
            data.rename(columns=rename_dict, inplace=True)
            data[date_key] = pd.to_datetime(data[date_key])

            if len(symbol_list) == len(path_list):
                data[symbol_key] = symbol_list[i]

            if symbol_key in data.columns:
                data[symbol_key] = data[symbol_key].astype(str)
            if begin is not None and end is not None:
                data = data.query(f"{date_key} >= '{begin}' and {date_key} <= '{end}'")

            data.set_index(date_key, inplace=True)
            data.index = data.index.map(lambda x: np.datetime64(x.date()))
            data_list.append(data)

        fina_res = data_list
        if not isinstance(data_path_list, list):
            fina_res = data_list[0]
        return fina_res
    
    @staticmethod
    def read_csv_from_tradingview(data_path_list: list, begin: str = None, end: str = None):
        """Reads a CSV file and filters the data by date range.

        Args:
            data_path (str): The path to the CSV file.
            begin (str): The start date for filtering (inclusive).
            end (str): The end date for filtering (inclusive).

        Returns:
            pd.DataFrame: The filtered data as a DataFrame.
        """
        path_list = data_path_list

        if isinstance(path_list, str):
            path_list = [path_list]

        rename_dict = {'datetime':'date', 'symbol':'code'}
        date_key, symbol_key = 'date', 'code'

        data_list = []
        for data_path in path_list:

            data = pd.read_csv(data_path, low_memory=True, encoding='utf_8_sig')
            data.rename(columns=rename_dict, inplace=True)
            data[date_key] = pd.to_datetime(data[date_key])

            if symbol_key in data.columns:
                data[symbol_key] = data[symbol_key].astype(str)
            if begin is not None and end is not None:
                data = data.query(f"{date_key} >= '{begin}' and {date_key} <= '{end}'")
                print(data)
            data.set_index(date_key, inplace=True)
            data.index = data.index.map(lambda x: np.datetime64(x.date()))
            # print(data)
            data_list.append(data)

        fina_res = data_list
        if isinstance(data_path_list, str):
            fina_res = data_list[0]
        return fina_res

    @staticmethod
    def read_csv(data_path: str, begin: str = None, end: str = None):
        data = pd.read_csv(data_path, parse_dates=['date'], dtype={'code': str}, low_memory=True, encoding='utf_8_sig')
        if begin is None and end is None:
            return data
        data = data.query(f"date >= '{begin}' and date <= '{end}'")
        return data
    
    @staticmethod
    def to_csv(data: pd.DataFrame, file_path: str):
        """Saves a DataFrame to a CSV file after renaming columns.

        Args:
            data (pd.DataFrame): The DataFrame to save.
            file_path (str): The path to save the CSV file.

        Returns:
            None
        """
        data.reset_index(drop=False, inplace=True)
        data.rename(columns={'datetime': 'date'}, inplace=True)
        data.rename(columns={'symbol': 'code'}, inplace=True)
        data.to_csv(file_path, encoding='utf_8_sig', index=False)

    @staticmethod
    def history_ohlc(symbol: str = None, start: str = None, end: str = None, adjust: str = 'qfq',
                     engine: str = 'tdx', username: str = None, password: str = None, exchange: str = None):
        """Fetches historical OHLC data for a given stock symbol.

        Args:
            symbol (str): The stock symbol to query.
            start (str): The start date for the data.
            end (str): The end date for the data.
            adjust (str): The adjustment method for the data (e.g., 'qfq').
            engine (str): The data source engine ('yahoo', 'tdx', or 'tv').
            username (str): The username for the TV data feed (if applicable).
            password (str): The password for the TV data feed (if applicable).
            exchange (str): The exchange for the TV data feed (if applicable).

        Returns:
            pd.DataFrame: The historical OHLC data as a DataFrame.
        """

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

        if engine == 'tv':
            try:
                tv = TvDatafeed(username, password)
                data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=10000)
                data.index = data.index.map(lambda x: np.datetime64(x.date()))
            except:
                data = None

        return data

    @staticmethod
    def _calc_func(func, m, kwargs={}):
        """Calculates a function and measures its execution time.

        Args:
            func (callable): The function to execute.
            m (str): The name of the function for logging purposes.
            kwargs (dict): The keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        t1 = time.time()
        alpha = func(**kwargs)
        t2 = time.time()
        print(f"{m} time {t2 - t1}")
        return alpha

    @staticmethod
    def tec_indicator(stock_data: pd.DataFrame, alpha_name: list = None, **kwargs):
        """Calculates technical indicators for the given stock data.

        Args:
            stock_data (pd.DataFrame): The stock data to analyze.
            alpha_name (list): A list of specific indicator names to calculate.
            **kwargs: Additional keyword arguments for the indicator functions.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated indicators.
        """
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
        """Calculates Alpha 101 factors for the given stock data.

        Args:
            stock_data (pd.DataFrame): The stock data to analyze.
            alpha_name (list): A list of specific alpha names to calculate.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated Alpha 101 factors.
        """

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
        """Calculates Alpha 191 factors for the given stock data against a benchmark.

        Args:
            stock_data (pd.DataFrame): The stock data to analyze.
            bench_data (pd.DataFrame): The benchmark data to compare against.
            alpha_name (list): A list of specific alpha names to calculate.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated Alpha 191 factors.
        """
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
