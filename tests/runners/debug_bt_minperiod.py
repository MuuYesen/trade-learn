"""Check Backtrader's actual min_period for Turtle."""
import pandas as pd
import backtrader as bt

DATA_PATH = "tests/data/AAPL.parquet"
df = pd.read_parquet(DATA_PATH)

class BTTurtleSpy(bt.Strategy):
    def __init__(self):
        hi, lo = self.data.high, self.data.low
        hi, lo = hi(-1), lo(-1)
        self.dch = bt.ind.Highest(hi, period=20)
        self.dcl = bt.ind.Lowest(lo, period=20)
        self.prenext_count = 0
        self.next_count = 0

    def prenext(self):
        self.prenext_count += 1
        
    def next(self):
        self.next_count += 1
        if self.next_count <= 3:
            print(f"  next() #{self.next_count} (after {self.prenext_count} prenexts), Close={self.data.close[0]:.2f}, DCH={self.dch[0]:.2f}, DCL={self.dcl[0]:.2f}")

cerebro = bt.Cerebro()
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(BTTurtleSpy)
cerebro.run()
print(f"Total prenext calls: {cerebro.runstrats[0][0].prenext_count}")
print(f"Total next calls: {cerebro.runstrats[0][0].next_count}")
