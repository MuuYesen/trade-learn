"""Check what min_period is being computed for each strategy."""
import pandas as pd
import tradelearn.compat.backtrader as tl
from examples import SmaCross, Turtle, EnhancedRSI

DATA_PATH = "tests/data/AAPL.parquet"
df = pd.read_parquet(DATA_PATH)

for name, cls in [("SmaCross", SmaCross), ("Turtle", Turtle), ("EnhancedRSI", EnhancedRSI)]:
    cerebro = tl.Cerebro()
    data = tl.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(cls)
    strategies = cerebro.run()
    print(f"{name}: _min_period = {strategies[0]._min_period}")
