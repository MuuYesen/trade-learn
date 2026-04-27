
import tradelearn.compat.backtrader as tl
import pandas as pd

class QuickstartSmaCross(tl.Strategy):
    def __init__(self):
        self.fast = tl.indicators.SMA(self.data.close, period=3)
        self.slow = tl.indicators.SMA(self.data.close, period=6)
    def next(self):
        dt = str(self.data.datetime.date(0))
        if "2024-02-18" in dt:
            print(f"DEBUG: {dt} pos={self.getposition().size}, fast={self.fast[0]:.6f}, slow={self.slow[0]:.6f}")
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=10)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()

def run_tl():
    df = pd.read_parquet('tests/data/AAPL.parquet')
    cerebro = tl.Cerebro()
    cerebro.adddata(df)
    cerebro.addstrategy(QuickstartSmaCross)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()

if __name__ == "__main__":
    run_tl()
