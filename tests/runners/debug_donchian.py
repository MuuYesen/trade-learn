"""Compare DonchianChannels values between TradeLearn and Backtrader."""
import pandas as pd
import sys

DATA_PATH = "tests/data/AAPL.parquet"
df = pd.read_parquet(DATA_PATH)

# --- TradeLearn side ---
print("=== TradeLearn DonchianChannels ===")
import tradelearn.compat.backtrader as tl

cerebro = tl.Cerebro()
data = tl.feeds.PandasData(dataname=df)
cerebro.adddata(data)

class TLSpy(tl.Strategy):
    def __init__(self):
        from examples import Turtle
        # Recreate only the channel logic
        hi, lo = self.data.high, self.data.low
        hi, lo = hi(-1), lo(-1)
        self.dch = tl.ind.Highest(hi, period=20)
        self.dcl = tl.ind.Lowest(lo, period=20)
        self.count = 0

    def next(self):
        self.count += 1
        if 25 <= self.count <= 35:
            print(f"  Bar {self.count}: Close={self.data.close[0]:.2f}, DCH={self.dch[0]:.2f}, DCL={self.dcl[0]:.2f}")

cerebro.addstrategy(TLSpy)
cerebro.run()

# --- Backtrader side ---
print("\n=== Backtrader DonchianChannels ===")
import backtrader as bt

cerebro2 = bt.Cerebro()
data2 = bt.feeds.PandasData(dataname=df)
cerebro2.adddata(data2)

class BTSpy(bt.Strategy):
    def __init__(self):
        hi, lo = self.data.high, self.data.low
        hi, lo = hi(-1), lo(-1)
        self.dch = bt.ind.Highest(hi, period=20)
        self.dcl = bt.ind.Lowest(lo, period=20)
        self.count = 0

    def next(self):
        self.count += 1
        if 25 <= self.count <= 35:
            print(f"  Bar {self.count}: Close={self.data.close[0]:.2f}, DCH={self.dch[0]:.2f}, DCL={self.dcl[0]:.2f}")

cerebro2.addstrategy(BTSpy)
cerebro2.run()
