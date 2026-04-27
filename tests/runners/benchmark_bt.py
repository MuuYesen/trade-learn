import sys
import time
import importlib
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "tests" / "data" / "AAPL.parquet"

# Strategies to benchmark
TARGET_STRATEGIES = [
    "01_quickstart",
    "02_sma_cross",
    "05_migration",
    "06_turtle",
    "07_rsi_enhanced",
    "08_better_ma",
    "09_macd_settings",
    "10_order_execution",
]

def get_strategy_class(module_name: str, class_name: str):
    module = importlib.import_module(f"examples.{module_name}")
    importlib.reload(module)  # Ensure it picks up the current sys.modules mock
    return getattr(module, class_name)

STRATEGY_CLASSES = {
    "01_quickstart": "QuickstartSmaCross",
    "02_sma_cross": "SmaCross",
    "05_migration": "MigratedSmaCross",
    "06_turtle": "Turtle",
    "07_rsi_enhanced": "EnhancedRSI",
    "08_better_ma": "BetterMA",
    "09_macd_settings": "MacdTharp",
    "10_order_execution": "OrderExecutionStrategy",
}

def load_data(bt_module):
    dataframe = pd.read_parquet(DATA_PATH)
    if hasattr(bt_module, 'feeds'):

        return bt_module.feeds.PandasData(dataname=dataframe)
    else:
        # tradelearn compat 
        return dataframe

def run_benchmark(engine_name: str, bt_module):
    print(f"\n{'='*40}")
    print(f"Running Benchmark with {engine_name.upper()}")
    print(f"{'='*40}")
    
    results = {}
    
    for mod_name in TARGET_STRATEGIES:
        cls_name = STRATEGY_CLASSES[mod_name]
        try:
            strategy_cls = get_strategy_class(mod_name, cls_name)
            print(f"[{cls_name}] MRO: {[c.__name__ for c in strategy_cls.mro()]}")
            
            # Setup Cerebro
            cerebro = bt_module.Cerebro()
            cerebro.broker.setcash(100000.0)
            
            data = load_data(bt_module)
            cerebro.adddata(data)
            cerebro.addstrategy(strategy_cls)
            
            start_time = time.perf_counter()
            strats = cerebro.run()
            end_time = time.perf_counter()
            
            strategy = strats[0]
            final_value = cerebro.broker.getvalue()
            final_cash = cerebro.broker.getcash()
            # Get position for first data
            pos = strategy.getposition(strategy.datas[0])
            print(f"[{cls_name}] Final Cash: {final_cash:.2f} | Pos: {pos.size} @ {pos.price:.2f}")
            elapsed_ms = (end_time - start_time) * 1000
            
            results[cls_name] = {
                "Time (ms)": elapsed_ms,
                "Final Value": final_value
            }
            print(f"[{cls_name}] Value: {final_value:.2f} | Time: {elapsed_ms:.2f} ms")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{cls_name}] FAILED: {e}")
            results[cls_name] = {"Time (ms)": None, "Final Value": None}

            
    return results

def main():
    try:
        import backtrader as real_bt
        has_real_bt = True
    except ImportError:
        has_real_bt = False

    if not has_real_bt:
        print("Error: The original 'backtrader' package is not installed.")
        print("Please run: pip install backtrader")
        return

    import tradelearn.compat.backtrader as tl_bt

    # 1. Run with Tradelearn Compat
    tl_results = run_benchmark("Tradelearn", tl_bt)

    # 2. Run with Real Backtrader
    # Mock the import so strategies load real backtrader
    original_attrs = {k: getattr(tl_bt, k) for k in dir(tl_bt)}
    
    # Overwrite tl_bt attributes with real_bt attributes
    for k in dir(real_bt):
        if not k.startswith("__"):
            setattr(tl_bt, k, getattr(real_bt, k))
            
    real_results = run_benchmark("Original Backtrader", real_bt)
    
    # Restore mock
    for k in dir(tl_bt):
        if not k.startswith("__"):
            delattr(tl_bt, k)
    for k, v in original_attrs.items():
        setattr(tl_bt, k, v)


    # 3. Compare Results
    print(f"\n{'='*40}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*40}")
    
    for mod_name in TARGET_STRATEGIES:
        cls_name = STRATEGY_CLASSES[mod_name]
        tl_res = tl_results.get(cls_name)
        real_res = real_results.get(cls_name)
        
        if tl_res and real_res and tl_res["Final Value"] is not None and real_res["Final Value"] is not None:
            tl_time = tl_res["Time (ms)"]
            real_time = real_res["Time (ms)"]
            speedup = real_time / tl_time if tl_time > 0 else 0
            
            val_diff = tl_res["Final Value"] - real_res["Final Value"]
            match_status = "✅ EXACT MATCH" if abs(val_diff) < 0.01 else f"❌ DIFF: {val_diff:.2f}"
            
            print(f"--- {cls_name} ---")
            print(f"  Tradelearn: {tl_res['Final Value']:.2f} ({tl_res['Time (ms)']:.2f} ms)")
            print(f"  Backtrader: {real_res['Final Value']:.2f} ({real_res['Time (ms)']:.2f} ms)")
            print(f"  Value Match: {match_status}")
            print(f"  Speedup: Tradelearn is {speedup:.2f}x faster")
        else:
            print(f"--- {cls_name} ---")
            print("  Could not compare due to execution failure.")

if __name__ == "__main__":
    main()
