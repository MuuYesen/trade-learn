# 1. Load the original backtesting.py
# We need to bypass the local shim 'backtesting.py'
import argparse
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

import tradelearn.compat.backtesting as tl_bt


def load_original_backtesting():
    # Remove current directory from path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = sys.path[:]
    sys.path = [p for p in sys.path if p != current_dir and p != ""]

    import backtesting as bt_orig

    importlib.reload(bt_orig)  # Ensure it's the real one

    # Restore path
    sys.path = orig_path
    return bt_orig


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data" / "backtesting"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT")


# 3. Define the Strategy (same code for both)
def get_strategy_class(BaseClass):
    class EMA_Cross_Strategy(BaseClass):
        ema_fast = 9
        ema_slow = 21
        trailing_pct = 0.03
        position_pct = 0.95

        def init(self):
            close = pd.Series(self.data.Close)
            self.ema9_ind = self.I(lambda: close.ewm(span=self.ema_fast, adjust=False).mean())
            self.ema21_ind = self.I(lambda: close.ewm(span=self.ema_slow, adjust=False).mean())
            self.highest_since_entry = 0

        def next(self):
            price = self.data.Close[-1]
            ema9 = self.ema9_ind[-1]
            ema21 = self.ema21_ind[-1]
            ema9_prev = self.ema9_ind[-2] if len(self.ema9_ind) > 1 else ema9
            ema21_prev = self.ema21_ind[-2] if len(self.ema21_ind) > 1 else ema21

            if self.position:
                if price > self.highest_since_entry:
                    self.highest_since_entry = price

                trailing_stop = self.highest_since_entry * (1 - self.trailing_pct)

                if price < trailing_stop:
                    self.position.close()
                    self.highest_since_entry = 0
                    return

                if ema9_prev >= ema21_prev and ema9 < ema21:
                    self.position.close()
                    self.highest_since_entry = 0
                    return
            else:
                if ema9_prev <= ema21_prev and ema9 > ema21:
                    self.buy(size=self.position_pct)
                    self.highest_since_entry = price

    return EMA_Cross_Strategy


# 4. Data Loading
def load_data(symbol):
    filepath = DATA_DIR / f"{symbol}_30m.csv"
    df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    if symbol == "BTCUSDT":
        scale_factor = 1000
        df["Open"] /= scale_factor
        df["High"] /= scale_factor
        df["Low"] /= scale_factor
        df["Close"] /= scale_factor
        df["Volume"] *= scale_factor
    return df


def _timed_run(factory, *, warmup: int, repeat: int):
    for _ in range(warmup):
        factory()
    durations = []
    stats = None
    for _ in range(repeat):
        start_time = time.time()
        stats = factory()
        durations.append(time.time() - start_time)
    return stats, statistics.median(durations)


def _comparison_passed(
    results: dict[str, dict[str, float]],
    *,
    min_speedup: float,
    return_tol: float = 1e-2,
    trades_tol: float = 0.0,
) -> bool:
    for result in results.values():
        if abs(result["return_diff"]) > return_tol:
            return False
        if abs(result["trades_diff"]) > trades_tol:
            return False
        if result["speedup"] < min_speedup:
            return False
    return True


def run_comparison(symbol, *, warmup: int = 0, repeat: int = 1):
    print(f"\nComparing results for {symbol}...")
    data = load_data(symbol)
    bar_count = len(data)
    initial_cash = 500000 if symbol == "BTCUSDT" else 5000

    # Load Original backtesting.py
    bt_orig = load_original_backtesting()
    orig_strat_cls = get_strategy_class(bt_orig.Strategy)

    print("Running Original backtesting.py...")

    def run_original():
        bt = bt_orig.Backtest(
            data,
            orig_strat_cls,
            cash=initial_cash,
            commission=0.0008,
            exclusive_orders=True,
        )
        return bt.run()

    stats_orig, orig_duration = _timed_run(run_original, warmup=warmup, repeat=repeat)

    # Run Tradelearn
    tl_strat_cls = get_strategy_class(tl_bt.Strategy)

    print("Running Tradelearn Facade...")

    def run_tradelearn():
        bt_tl = tl_bt.Backtest(
            data,
            tl_strat_cls,
            cash=initial_cash,
            commission=0.0008,
            exclusive_orders=True,
        )
        return bt_tl.run()

    stats_tl, tl_duration = _timed_run(run_tradelearn, warmup=warmup, repeat=repeat)

    # Compare
    print("\n" + "=" * 50)
    print(f"RESULTS COMPARISON: {symbol}")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Original':<12} | {'Tradelearn':<12} | {'Diff':<10}")
    print("-" * 65)

    metrics = [
        ("Return [%]", "Return [%]"),
        ("Equity Final [$]", "Equity Final [$]"),
        ("# Trades", "# Trades"),
        ("Win Rate [%]", "Win Rate [%]"),
    ]

    diffs = {}
    for label, key in metrics:
        v_orig = stats_orig[key]
        v_tl = stats_tl[key]
        diff = v_tl - v_orig
        diffs[key] = float(diff)
        print(f"{label:<20} | {v_orig:>12.2f} | {v_tl:>12.2f} | {diff:>10.2f}")

    print("-" * 65)
    orig_bars_per_sec = bar_count / orig_duration if orig_duration else 0.0
    tl_bars_per_sec = bar_count / tl_duration if tl_duration else 0.0
    bars_per_sec_speedup = tl_bars_per_sec / orig_bars_per_sec if orig_bars_per_sec else 0.0
    print(
        f"{'Duration [s]':<20} | {orig_duration:>12.4f} | "
        f"{tl_duration:>12.4f} | {orig_duration / tl_duration:>10.2f}x"
    )
    print(
        f"{'Bars/s':<20} | {orig_bars_per_sec:>12,.0f} | "
        f"{tl_bars_per_sec:>12,.0f} | {bars_per_sec_speedup:>10.2f}x"
    )
    print("=" * 50)
    return {
        "return_diff": diffs["Return [%]"],
        "trades_diff": diffs["# Trades"],
        "speedup": orig_duration / tl_duration if tl_duration else 0.0,
        "orig_duration": orig_duration,
        "tl_duration": tl_duration,
        "orig_bars_per_sec": orig_bars_per_sec,
        "tl_bars_per_sec": tl_bars_per_sec,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tradelearn with backtesting.py.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--return-tol", type=float, default=1e-2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = {
        symbol: run_comparison(symbol, warmup=args.warmup, repeat=args.repeat)
        for symbol in args.symbols
    }
    if args.min_speedup:
        passed = _comparison_passed(
            results,
            min_speedup=args.min_speedup,
            return_tol=args.return_tol,
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"\nGate: {status} "
            f"(min_speedup={args.min_speedup:.2f}x, return_tol={args.return_tol})"
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
