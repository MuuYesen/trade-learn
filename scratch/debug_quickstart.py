
import tradelearn.compat.backtrader as bt
import pandas as pd

class QuickstartSmaCross(bt.Strategy):
    params = (('pfast', 3), ('pslow', 6),)
    def __init__(self):
        self.sma1 = bt.indicators.SMA(period=self.p.pfast)
        self.sma2 = bt.indicators.SMA(period=self.p.pslow)
    def next(self):
        if not self.position:
            if self.sma1[0] > self.sma2[0]:
                self.buy(size=100)
        elif self.sma1[0] < self.sma2[0]:
            self.sell(size=100)

def run():
    df = pd.read_parquet('tests/data/AAPL.parquet')
    cerebro = bt.Cerebro()
    cerebro.adddata(df)
    cerebro.addstrategy(QuickstartSmaCross)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    strats = cerebro.run()
    print(f"Final Value: {cerebro.broker.getvalue()}")
    for fill in cerebro.broker.fills_frame().itertuples():
        print(f"{fill.datetime} {fill.side} {fill.size} @ {fill.price} Comm: {fill.commission:.4f}")

if __name__ == "__main__":
    run()
