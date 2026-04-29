import argparse
import importlib
import statistics
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import pandas as pd

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

PREVIOUS_TL_MS = {
    "QuickstartSmaCross": 6.4,
    "SmaCross": 5.6,
    "MigratedSmaCross": 6.5,
    "Turtle": 9.6,
    "EnhancedRSI": 8.1,
    "BetterMA": 5.3,
    "MacdTharp": 7.0,
    "OrderExecutionStrategy": 7.4,
}


def run_strategy_in_process(
    engine_type,
    mod_name,
    cls_name,
    queue,
    match_mode="exact",
    repeats=1,
    warmup=0,
):
    try:
        import contextlib
        import io
        import time
        import types

        if engine_type == "Tradelearn":
            import tradelearn.engine as bt
        else:
            import backtrader as bt

            sys.modules["tradelearn"] = types.ModuleType("tradelearn")
            sys.modules["tradelearn.engine"] = bt

        module = importlib.import_module(f"examples.backtrader.{mod_name}")
        importlib.reload(module)
        strategy_cls = getattr(module, cls_name)

        # Inject logging into the strategy class
        def notify_trade(self, trade):
            if trade.isclosed:
                self.audit_log.append(
                    {
                        "pnl": trade.pnl,
                        "pnlcomm": trade.pnlcomm,
                        "price": trade.price,
                        "size": trade.size,
                        "dt": self.data.datetime.date(0).isoformat(),
                    }
                )

        strategy_cls.notify_trade = notify_trade
        strategy_cls.audit_log = []

        dataframe = pd.read_parquet(DATA_PATH)
        if engine_type == "Tradelearn":
            from tradelearn.engine import DataFeed
        else:

            class PandasData(bt.feeds.PandasData):
                params = (
                    ("datetime", None),
                    ("open", "open"),
                    ("high", "high"),
                    ("low", "low"),
                    ("close", "close"),
                    ("volume", "volume"),
                    ("openinterest", None),
                )

        repeats = max(1, int(repeats))
        warmup = max(0, int(warmup))
        suppress_run_output = warmup > 0 or repeats > 1
        timings = []
        final_value = None
        trades = None
        for run_idx in range(warmup + repeats):
            if engine_type == "Tradelearn":
                cerebro = bt.Cerebro(match_mode=match_mode)
            else:
                cerebro = bt.Cerebro()
            cerebro.broker.setcash(100000.0)
            cerebro.broker.setcommission(commission=0.0)
            if engine_type == "Tradelearn":
                data = DataFeed(dataframe)
            else:
                data = PandasData(dataname=dataframe)
            cerebro.adddata(data)
            cerebro.addstrategy(strategy_cls)
            strategy_cls.audit_log = []

            start_time = time.perf_counter()
            if suppress_run_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    strats = cerebro.run()
            else:
                strats = cerebro.run()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if run_idx >= warmup:
                timings.append(elapsed_ms)
                final_value = cerebro.broker.getvalue()
                trades = list(strats[0].audit_log)

        elapsed_ms = statistics.median(timings)

        queue.put(
            {
                "status": "success",
                "final_value": final_value,
                "elapsed_ms": elapsed_ms,
                "trades": trades,
                "timings_ms": timings,
            }
        )
    except Exception as e:
        import traceback

        queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_benchmark(
    match_mode="smart",
    repeats: int = 1,
    warmup: int = 0,
    min_speedup: float = 0.0,
) -> bool:
    print(f"\n{'=' * 80}")
    timing_label = "median" if repeats > 1 else "single run"
    print(
        f"ULTIMATE NUMERICAL AUDIT: Tradelearn ({match_mode}) vs Backtrader "
        f"[{timing_label}, repeats={repeats}, warmup={warmup}]"
    )
    print(f"{'=' * 80}")

    final_results = {}

    for mod_name in TARGET_STRATEGIES:
        cls_name = STRATEGY_CLASSES[mod_name]
        print(f"\nAudit {cls_name} ...")

        results = {}
        for engine in ["Tradelearn", "Backtrader"]:
            queue = Queue()
            # Pass match_mode only to Tradelearn
            mode = match_mode if engine == "Tradelearn" else "exact"
            p = Process(
                target=run_strategy_in_process,
                args=(engine, mod_name, cls_name, queue, mode, repeats, warmup),
            )
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
                print("  [DIVERGENCE DETECTED]")
                # Compare trades to find the first split
                t1, t2 = tl["trades"], bt_res["trades"]
                for i in range(min(len(t1), len(t2))):
                    pnl_diff = abs(t1[i]["pnl"] - t2[i]["pnl"])
                    size_diff = abs(t1[i]["size"] - t2[i]["size"])
                    if pnl_diff > 1e-6 or size_diff > 1e-6:
                        print(f"  First trade split at {t1[i]['dt']}:")
                        print(
                            f"    TL: PnL={t1[i]['pnl']:.4f}, "
                            f"Size={t1[i]['size']:.4f}, Price={t1[i]['price']:.4f}"
                        )
                        print(
                            f"    BT: PnL={t2[i]['pnl']:.4f}, "
                            f"Size={t2[i]['size']:.4f}, Price={t2[i]['price']:.4f}"
                        )
                        break
                if len(t1) != len(t2):
                    print(f"  Trade count mismatch: TL={len(t1)}, BT={len(t2)}")

        final_results[cls_name] = results

    print(f"\n\n{'=' * 120}")
    comparable_to_previous = repeats == 1 and warmup == 0
    prev_header = "vs Prev TL" if comparable_to_previous else "vs Prev TL*"
    print(
        f"{'Strategy':<25} | {'TL Value':<12} | {'BT Value':<12} | "
        f"{'TL Time':<10} | {'BT Time':<10} | {'Speedup':<10} | "
        f"{prev_header:<11} | {'Status':<10}"
    )
    print(f"{'-' * 136}")
    for cls_name, res in final_results.items():
        tl, bt_res = res.get("Tradelearn"), res.get("Backtrader")
        if tl and bt_res:
            diff = tl["final_value"] - bt_res["final_value"]
            t_tl, t_bt = tl["elapsed_ms"], bt_res["elapsed_ms"]
            speedup = t_bt / t_tl if t_tl > 0 else 0
            prev_tl = PREVIOUS_TL_MS.get(cls_name)
            if comparable_to_previous and prev_tl:
                improvement = (prev_tl - t_tl) / prev_tl * 100
                improvement_text = f"{improvement:+6.1f}%"
            else:
                improvement_text = "warm run"
            exact = abs(diff) < EXACT_TOLERANCE
            fast_enough = min_speedup <= 0 or speedup >= min_speedup
            status = "✅ EXACT" if exact else "❌ DIFF"
            if exact and not fast_enough:
                status = f"❌ SLOW < {min_speedup:.1f}x"
            print(
                f"{cls_name:<25} | {tl['final_value']:<12.2f} | "
                f"{bt_res['final_value']:<12.2f} | {t_tl:>7.1f}ms | "
                f"{t_bt:>7.1f}ms | {speedup:>8.1f}x | "
                f"{improvement_text:>11} | {status:<10}"
            )
    print(f"{'=' * 136}\n")
    if not comparable_to_previous:
        print(
            "* Warm/repeated runs are not directly comparable with the saved single-run "
            "previous TL baseline.\n"
        )
    return _benchmark_passed(final_results, min_speedup=min_speedup)


def _benchmark_passed(final_results: dict, min_speedup: float = 0.0) -> bool:
    for res in final_results.values():
        tl, bt_res = res.get("Tradelearn"), res.get("Backtrader")
        if not tl or not bt_res:
            return False
        if abs(tl["final_value"] - bt_res["final_value"]) >= EXACT_TOLERANCE:
            return False
        if min_speedup > 0 and bt_res["elapsed_ms"] / tl["elapsed_ms"] < min_speedup:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="smart")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    args = parser.parse_args()
    ok = run_benchmark(
        args.mode,
        repeats=args.repeat,
        warmup=args.warmup,
        min_speedup=args.min_speedup,
    )
    raise SystemExit(0 if ok else 1)
