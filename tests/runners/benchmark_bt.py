import sys
import time
import importlib
import pandas as pd
from pathlib import Path
from multiprocessing import Process, Queue
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "tests" / "data" / "AAPL.parquet"

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

EXACT_TOLERANCE = 1e-3

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

def run_strategy_in_process(engine_type, mod_name, cls_name, queue, match_mode='exact'):
    try:
        import types
        import time
        if engine_type == "Tradelearn":
            import tradelearn.compat.backtrader as bt
        else:
            import backtrader as bt
            sys.modules['tradelearn'] = types.ModuleType("tradelearn")
            sys.modules['tradelearn.compat'] = types.ModuleType("tradelearn.compat")
            sys.modules['tradelearn.compat.backtrader'] = bt
        
        module = importlib.import_module(f"examples.backtrader.{mod_name}")
        importlib.reload(module)
        strategy_cls = getattr(module, cls_name)
        
        # Inject logging into the strategy class
        def notify_trade(self, trade):
            if trade.isclosed:
                self.audit_log.append({
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "price": trade.price,
                    "size": trade.size,
                    "dt": self.data.datetime.date(0).isoformat()
                })
        
        strategy_cls.notify_trade = notify_trade
        strategy_cls.audit_log = []

        if engine_type == "Tradelearn":
            cerebro = bt.Cerebro(match_mode=match_mode)
        else:
            cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.0)
        
        dataframe = pd.read_parquet(DATA_PATH)
        if engine_type == "Tradelearn":
            from tradelearn.compat.backtrader import DataFeed
            data = DataFeed(dataframe)
        else:
            class PandasData(bt.feeds.PandasData):
                params = (('datetime', None), ('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'), ('volume', 'volume'), ('openinterest', None))
            data = PandasData(dataname=dataframe)
        
        cerebro.adddata(data)
        cerebro.addstrategy(strategy_cls)
        
        start_time = time.perf_counter()
        strats = cerebro.run()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        queue.put({
            "status": "success",
            "final_value": cerebro.broker.getvalue(),
            "elapsed_ms": elapsed_ms,
            "trades": strats[0].audit_log
        })
    except Exception as e:
        import traceback
        queue.put({
            "status": "error", "error": str(e), "traceback": traceback.format_exc()
        })

def run_benchmark(match_mode='smart'):
    print(f"\n{'='*80}")
    print(f"ULTIMATE NUMERICAL AUDIT: Tradelearn ({match_mode}) vs Backtrader")
    print(f"{'='*80}")
    
    final_results = {}

    for mod_name in TARGET_STRATEGIES:
        cls_name = STRATEGY_CLASSES[mod_name]
        print(f"\nAudit {cls_name} ...")
        
        results = {}
        for engine in ["Tradelearn", "Backtrader"]:
            queue = Queue()
            # Pass match_mode only to Tradelearn
            mode = match_mode if engine == "Tradelearn" else "exact"
            p = Process(target=run_strategy_in_process, args=(engine, mod_name, cls_name, queue, mode))
            p.start()
            res = queue.get()
            p.join()
            if res["status"] == "success":
                results[engine] = res
            else:
                print(f"  [{engine}] FAILED: {res['error']}")
                results[engine] = None
        
        tl, bt_res = results.get("Tradelearn"), results.get("Backtrader")
        if tl and bt_res:
            diff = tl["final_value"] - bt_res["final_value"]
            status = "✅ EXACT" if abs(diff) < EXACT_TOLERANCE else f"❌ DIFF: {diff:.4f}"
            print(f"  Result: {tl['final_value']:.2f} vs {bt_res['final_value']:.2f} | {status}")
            
            if abs(diff) >= EXACT_TOLERANCE:
                print(f"  [DIVERGENCE DETECTED]")
                # Compare trades to find the first split
                t1, t2 = tl["trades"], bt_res["trades"]
                for i in range(min(len(t1), len(t2))):
                    if abs(t1[i]["pnl"] - t2[i]["pnl"]) > 1e-6 or abs(t1[i]["size"] - t2[i]["size"]) > 1e-6:
                        print(f"  First trade split at {t1[i]['dt']}:")
                        print(f"    TL: PnL={t1[i]['pnl']:.4f}, Size={t1[i]['size']:.4f}, Price={t1[i]['price']:.4f}")
                        print(f"    BT: PnL={t2[i]['pnl']:.4f}, Size={t2[i]['size']:.4f}, Price={t2[i]['price']:.4f}")
                        break
                if len(t1) != len(t2):
                    print(f"  Trade count mismatch: TL={len(t1)}, BT={len(t2)}")
        
        final_results[cls_name] = results

    print(f"\n\n{'='*120}")
    print(f"{'Strategy':<25} | {'TL Value':<12} | {'BT Value':<12} | {'TL Time':<10} | {'BT Time':<10} | {'Speedup':<10} | {'Status':<10}")
    print(f"{'-'*120}")
    for cls_name, res in final_results.items():
        tl, bt_res = res.get("Tradelearn"), res.get("Backtrader")
        if tl and bt_res:
            diff = tl["final_value"] - bt_res["final_value"]
            t_tl, t_bt = tl["elapsed_ms"], bt_res["elapsed_ms"]
            speedup = t_bt / t_tl if t_tl > 0 else 0
            status = "✅ EXACT" if abs(diff) < EXACT_TOLERANCE else "❌ DIFF"
            print(f"{cls_name:<25} | {tl['final_value']:<12.2f} | {bt_res['final_value']:<12.2f} | {t_tl:>7.1f}ms | {t_bt:>7.1f}ms | {speedup:>8.1f}x | {status:<10}")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smart'
    run_benchmark(mode)
