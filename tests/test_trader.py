import unittest
from tradelearn.trader import Backtest, Strategy
from tradelearn.trader.lib import crossover
from tradelearn.query.query import Query

class TestCuasal(unittest.TestCase):

    def test_trader_backtest(self):
        import numpy as np
        import pandas as pd

        def _read_file(filename):
            from os.path import dirname, join
            return pd.read_csv(join(dirname(__file__), filename), index_col=0, parse_dates=True)

        # GOOG = _read_file('data/GOOG.csv')

        GOOG = Query.history_ohlc(symbol='GOOG', engine='tv', username='muyes88', password='toxka2-ternez-cakZah', exchange='NASDAQ')
        GOOG.index = GOOG.index.map(lambda x: np.datetime64(x.date()))

        def SMA(arr: pd.Series, n: int) -> pd.Series:
            return arr.rolling(n).mean()

        class SmaCross(Strategy):
            fast = 10
            slow = 20

            def init(self):
                price = self.data.close.df  # 2
                self.ma1 = self.I(SMA, price, self.fast, overlay=True)  # 3
                self.ma2 = self.I(SMA, price, self.slow, overlay=True)  # 3

            def next(self):
                if crossover(self.ma1, self.ma2):
                    self.position().close()  # 4
                    self.buy()
                elif crossover(self.ma2, self.ma1):
                    self.position().close()  # 4
                    self.sell()

        bt = Backtest(GOOG, SmaCross, cash=1000000, commission=.002, trade_on_close=False)
        stats = bt.run()
        print(stats)
        bt.plot(plot_volume=True, superimpose=True)