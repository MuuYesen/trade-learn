import unittest
from tradelearn.trader import Backtest, Strategy
from tradelearn.trader.libs.lib import crossover
from tradelearn.query.query import Query

class TestCuasal(unittest.TestCase):

    def test_trader_backtest(self):
        GOOG = Query.history_ohlc(engine='tv', symbol='GOOG', exchange='NASDAQ')

        class SmaCross(Strategy):
            fast = 10
            slow = 20

            def init(self):
                def SMA(arr, n):
                    return arr.rolling(n).mean()

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