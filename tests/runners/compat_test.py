"""
Generic Runner for verifying Backtrader strategy compatibility.
Usage: PYTHONPATH=. python tests/runners/compat_test.py <strategy_name>
Example: PYTHONPATH=. python tests/runners/compat_test.py TurtleStrategy
"""

import sys
import pandas as pd
import tradelearn.compat.backtrader as bt
from examples import (
    SmaCross, 
    Alpha101GBMStrategy, 
    RandomForestRotation, 
    QuickstartSmaCross, 
    MigratedSmaCross,
    Turtle,
    EnhancedRSI,
    BetterMA,
    MacdTharp,
    OrderExecutionStrategy
)

# Map of strategy names to classes (focused on Backtrader compatibility)
STRATEGIES = {
    "SmaCross": SmaCross,
    "QuickstartSmaCross": QuickstartSmaCross,
    "MigratedSmaCross": MigratedSmaCross,

    "Turtle": Turtle,
    "EnhancedRSI": EnhancedRSI,
    "BetterMA": BetterMA,
    "MacdTharp": MacdTharp,
    "OrderExecutionStrategy": OrderExecutionStrategy,
}

def run_compat_test(strategy_name: str):
    if strategy_name not in STRATEGIES:
        print(f"Error: Strategy '{strategy_name}' not found.")
        print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        return

    print(f"--- Running Compatibility Test: {strategy_name} ---")
    
    # 1. Setup Cerebro
    cerebro = bt.Cerebro()
    
    # 2. Add Data
    try:
        df = pd.read_parquet("tests/data/AAPL.parquet")
        data = bt.feeds.PandasData(dataname=df, name="AAPL")
        cerebro.adddata(data)
        
        # Some strategies (like EnhancedRSI) might expect a second data feed
        if strategy_name == "EnhancedRSI":
            df2 = pd.read_parquet("tests/data/GOOG.parquet")
            data2 = bt.feeds.PandasData(dataname=df2, name="GOOG")
            cerebro.adddata(data2)
    except FileNotFoundError:
        print("Error: tests/data/AAPL.parquet not found. Please ensure data is in tests/data/.")
        return

    # 3. Add Strategy
    cerebro.addstrategy(STRATEGIES[strategy_name])
    
    # 4. Configure Broker
    cerebro.broker.setcash(100000.0)
    
    # 5. Run
    print("Starting Backtest...")
    cerebro.run()
    
    # 6. Result
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {final_value:.2f}")
    print("--- Test Completed Successfully ---\n")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Run specific strategy if argument provided
        run_compat_test(sys.argv[1])
    else:
        # Default: Run all strategies
        print("No strategy specified. Running full compatibility sweep...\n")
        success_count = 0
        for name in STRATEGIES.keys():
            try:
                run_compat_test(name)
                success_count += 1
            except Exception as e:
                print(f"FAILED: {name} - Error: {e}\n")
        
        print(f"Full Sweep Completed: {success_count}/{len(STRATEGIES)} strategies passed.")

