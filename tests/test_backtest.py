import unittest
from tradelearn.strategy.backtest import Backtest, Strategy
from tradelearn.strategy.backtest.backtesting_py.libs.lib import crossover
from tradelearn.query.query import Query

class TestBackTest(unittest.TestCase):

    def test_trader_backtest(self):
        GOOG = Query.history_ohlc(engine='tv', symbol='GOOG', exchange='NASDAQ')

        class SmaCross(Strategy):
            fast = 10
            slow = 20

            def init(self):
                def SMA(arr, n):
                    return arr.rolling(n).mean()

                price = self.data.close.df
                self.ma1 = self.I(SMA, price, self.fast, overlay=True)
                self.ma2 = self.I(SMA, price, self.slow, overlay=True)

            def next(self):
                if crossover(self.ma1, self.ma2):
                    self.position().close()
                    self.buy()
                elif crossover(self.ma2, self.ma1):
                    self.position().close()
                    self.sell()

        bt = Backtest(GOOG, SmaCross, cash=1000000, commission=.002, trade_on_close=False)
        stats = bt.run()
        print(stats)
        bt.plot(plot_volume=True, superimpose=True)


    def test_trader_portfolio(self):
        class MyStrategy(Strategy):
            lookback = 10

            def init(self):
                self.roc = self.I(self.data.ta.roc(self.lookback), name='ROC')  # 1

            def next(self):
                self.alloc.assume_zero()  # 2
                roc = self.roc.df.iloc[-1]  # 3
                (self.alloc.bucket['equity']  # 4
                 .append(roc.sort_values(ascending=False), roc > 0)  # 5
                 .trim(3)  # 6
                 .weight_explicitly(1 / 3)  # 7
                 .apply())  # 8
                self.rebalance(cash_reserve=0.01)  # 9

        GOOG = Query.history_ohlc(engine='tv', symbol='GOOG', exchange='NASDAQ')

        data = yahoo.daily_bar('MMM,AXP,AAPL,BA,CVX', start='2023-01-01')
        bt = Backtest(data, MyStrategy, cash=10000)
        result = bt.run()
        print(result)
        bt.plot(plot_allocation=True)
