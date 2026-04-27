import pandas as pd
import numpy as np
import tradelearn.compat.backtrader as tl
import backtrader as bt
from examples import SmaCross

def run_tl():
    print("\n--- TradeLearn Results ---")
    df = pd.read_parquet("tests/data/AAPL.parquet")
    cerebro = tl.Cerebro()
    cerebro.broker.setcash(100000.0)
    data = tl.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    class DebugSmaCross(SmaCross):
        def notify_trade(self, trade):
            if trade.isclosed:
                print(f"Date: {self.data.datetime.date(0)}, PnL: {trade.pnl:.2f}, PnLComm: {trade.pnlcomm:.2f}")

    cerebro.addstrategy(DebugSmaCross)
    cerebro.run()
    print(f"Final Value: {cerebro.broker.getvalue():.2f}")

def run_bt():
    print("\n--- Backtrader Results ---")
    df = pd.read_parquet("tests/data/AAPL.parquet")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    class DebugSmaCross(SmaCross):
        def notify_trade(self, trade):
            if trade.isclosed:
                print(f"Date: {self.data.datetime.date(0)}, PnL: {trade.pnl:.2f}, PnLComm: {trade.pnlcomm:.2f}")

    cerebro.addstrategy(DebugSmaCross)
    cerebro.run()
    print(f"Final Value: {cerebro.broker.getvalue():.2f}")

if __name__ == "__main__":
    run_tl()
    run_bt()
