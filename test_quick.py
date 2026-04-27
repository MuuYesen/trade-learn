import sys
import importlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "tests" / "data" / "AAPL.parquet"

import tradelearn.compat.backtrader as tl_bt

module = importlib.import_module("examples.01_quickstart")
QuickstartSmaCross = getattr(module, "QuickstartSmaCross")

cerebro = tl_bt.Cerebro()
cerebro.broker.setcash(100000.0)

dataframe = pd.read_parquet(DATA_PATH)
data = dataframe

cerebro.adddata(data)
cerebro.addstrategy(QuickstartSmaCross)

cerebro.run()
print(cerebro.broker.getvalue())
