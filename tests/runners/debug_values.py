import pandas as pd
import numpy as np
import tradelearn.compat.backtrader as tl
from examples import SmaCross

def debug_values():
    df = pd.read_parquet("tests/data/AAPL.parquet")
    cerebro = tl.Cerebro()
    data = tl.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    class ValueSpy(SmaCross):
        def __init__(self):
            super().__init__()
            self.count = 0
        def next(self):
            self.count += 1
            if self.count < 40:
                print(f"Step {self.count}: Close={self.data.close[0]:.2f}, Fast={self.ma_fast[0]:.2f}, Slow={self.ma_slow[0]:.2f}")
            super().next()

    cerebro.addstrategy(ValueSpy)
    cerebro.run()

if __name__ == "__main__":
    debug_values()
