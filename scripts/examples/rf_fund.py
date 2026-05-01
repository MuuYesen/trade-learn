"""Runnable example for Portfolio Rotation Strategy using multiple assets."""

import pandas as pd

import tradelearn.engine as bt
from examples.engine import RandomForestRotation


def main():
    # 1. Load Multi-Asset Data
    try:
        aapl = pd.read_parquet("tests/data/AAPL.parquet")
        goog = pd.read_parquet("tests/data/GOOG.parquet")
        tsla = pd.read_parquet("tests/data/TSLA.parquet")
    except FileNotFoundError:
        print("Demo data not found. Run 'python scripts/generate_demo_data.py' first.")
        return

    # 2. Setup Cerebro
    cerebro = bt.Cerebro(trade_on_close=True)
    cerebro.broker.setcash(1000000.0)

    # 3. Add Multiple Data Feeds
    cerebro.adddata(bt.feeds.PandasData(dataname=aapl, name="AAPL"))
    cerebro.adddata(bt.feeds.PandasData(dataname=goog, name="GOOG"))
    cerebro.adddata(bt.feeds.PandasData(dataname=tsla, name="TSLA"))

    # 4. Add Rotation Strategy
    cerebro.addstrategy(RandomForestRotation, top_n=2, size=100)

    # 5. Run
    print("Starting Portfolio Rotation Backtest...")
    strategies = cerebro.run()
    result = strategies[0]

    if result.stats:
        print("\nPortfolio Backtest Summary:")
        print(result.stats.summary)


if __name__ == "__main__":
    main()
