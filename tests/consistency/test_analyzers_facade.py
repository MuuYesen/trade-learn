import pandas as pd
import numpy as np
import tradelearn.backtest as bt

def test_facade_analyzers():
    print("\n--- Testing Facade-based Metrics Engine ---")
    
    # 1. Create dummy data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    # create an upward trending price series
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.01, "low": prices * 0.99, "close": prices, "volume": 1000
    }, index=dates)

    # 2. Setup Cerebro
    cerebro = bt.Cerebro()
    data = bt.DataFeed(df)
    cerebro.adddata(data)
    
    # Simple strategy that just buys and holds
    class BuyAndHold(bt.Strategy):
        def next(self):
            if len(self) == 1:
                # Buy 100 shares to make a significant impact on a 10000 cash portfolio
                self.buy(size=50)
            if len(self) % 20 == 0:
                print(f"Bar {len(self)}: Portfolio Value = {self.broker.getvalue():.2f}")

    cerebro.addstrategy(BuyAndHold)
    
    # 3. Add Analyzers
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.Drawdown)
    
    # Add MULTIPLE instances of Sharpe with different parameters
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.01, _name="sharpe_1")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.05, _name="sharpe_5")

    # 4. Run Backtest
    print("Running cerebro (Vectorized mode)...")
    strats = cerebro.run()
    strat = strats[0]

    # 5. Extract Results
    print("\n--- Analysis Results ---")
    print("Returns:", strat.analyzers['returns'].get_analysis())
    print("Drawdown:", strat.analyzers['drawdown'].get_analysis())
    print("Sharpe (RF=0.01):", strat.analyzers['sharpe_1'].get_analysis())
    print("Sharpe (RF=0.05):", strat.analyzers['sharpe_5'].get_analysis())
    
    # Verify the instance ID concept
    engine = strat.metrics_engine
    print("\n--- Metrics Engine Internal State ---")
    print(f"Total Cached Results (Instance count): {len(engine.results)}")
    for inst_id, res in engine.results.items():
        print(f"[{inst_id}] -> {res}")

if __name__ == "__main__":
    test_facade_analyzers()
