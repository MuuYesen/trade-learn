from tradelearn.query import Query
from tradelearn.strategy.backtest import Backtest, Strategy
from tradelearn.strategy.evaluate import Evaluate

if __name__ == '__main__':

    GOOG = Query.history_ohlc(engine='tv', symbol='GOOG', exchange='NASDAQ')

    def crossover(series1, series2):
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]

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

    Evaluate.analysis_report(stats, GOOG, engine='quantstats')
    Evaluate.analysis_report(stats, GOOG, engine='pyfolio')

