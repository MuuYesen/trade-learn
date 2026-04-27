
import tradelearn.compat.backtrader as tl
import backtrader as bt
import pandas as pd
import numpy as np

def debug_sma():
    df = pd.read_parquet('tests/data/AAPL.parquet')
    
    # TradeLearn SMA
    sma3 = tl.indicators.SMA(df.close, period=3)
    sma6 = tl.indicators.SMA(df.close, period=6)
    
    # Backtrader SMA
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    class DebugStrat(bt.Strategy):
        def __init__(self):
            self.sma3 = bt.indicators.SMA(self.data.close, period=3)
            self.sma6 = bt.indicators.SMA(self.data.close, period=6)
        def next(self):
            dt = self.data.datetime.date(0)
            if str(dt) == "2024-02-18":
                print(f"BT 2024-02-18: sma3={self.sma3[0]:.12f}, sma6={self.sma6[0]:.12f}, diff={self.sma3[0]-self.sma6[0]:.12f}")
    cerebro.addstrategy(DebugStrat)
    cerebro.run()
    
    # TL Values
    idx = df.index.get_loc("2024-02-18")
    s3_val = sma3.lines.sma._values[idx]
    s6_val = sma6.lines.sma._values[idx]
    print(f"TL 2024-02-18: sma3={s3_val:.12f}, sma6={s6_val:.12f}, diff={s3_val-s6_val:.12f}")

if __name__ == "__main__":
    debug_sma()
