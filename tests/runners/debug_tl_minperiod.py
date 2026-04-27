"""Check TL's first next() call values for Turtle-like strategy."""
import pandas as pd
import tradelearn.compat.backtrader as tl

DATA_PATH = "tests/data/AAPL.parquet"
df = pd.read_parquet(DATA_PATH)

class TLSpy(tl.Strategy):
    def __init__(self):
        hi, lo = self.data.high, self.data.low
        hi, lo = hi(-1), lo(-1)
        self.dch = tl.ind.Highest(hi, period=20)
        self.dcl = tl.ind.Lowest(lo, period=20)
        self.next_count = 0

    def next(self):
        self.next_count += 1
        if self.next_count <= 3:
            print(f"  next() #{self.next_count}, Close={self.data.close[0]:.2f}, DCH={self.dch[0]:.2f}, DCL={self.dcl[0]:.2f}")

cerebro = tl.Cerebro()
data = tl.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(TLSpy)
strategies = cerebro.run()
print(f"_min_period = {strategies[0]._min_period}")
print(f"Total next calls: {strategies[0].next_count}")
