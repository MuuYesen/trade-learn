
import tradelearn.engine as bt
import pandas as pd
import numpy as np

class MultiTimeframeStrategy(bt.Strategy):
    def next(self):
        d0_close = self.datas[0].close[0]
        d1_close = self.datas[1].close[0] if len(self.datas[1]) > 0 else None
        
        d0_ts = self.datas[0].datetime[0]
        d1_ts = self.datas[1].datetime[0] if len(self.datas[1]) > 0 else None
        
        print(f"[{self.datas[0].datetime.date(0)}] d0_ts: {d0_ts}, d1_ts: {d1_ts}, d1: {d1_close}")

def test_resampling_alignment():
    # 1-minute data for 15 minutes
    dates = pd.date_range('2023-01-01 10:00', periods=15, freq='1min')
    df = pd.DataFrame({
        'open': np.linspace(100, 115, 15),
        'high': np.linspace(101, 116, 15),
        'low': np.linspace(99, 114, 15),
        'close': np.linspace(100.5, 115.5, 15),
        'volume': [100] * 15
    }, index=dates)

    cerebro = bt.Cerebro()
    data0 = bt.DataFeed(df, name="min1")
    cerebro.adddata(data0)
    
    # Use the new API
    data1 = cerebro.resampledata(data0, bt.TimeFrame.Minutes, 5)
    
    cerebro.addstrategy(MultiTimeframeStrategy)
    cerebro.run()

if __name__ == "__main__":
    test_resampling_alignment()
