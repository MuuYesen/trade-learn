# 1. Load the original backtesting.py
# We need to bypass the local shim 'backtesting.py'
import importlib.util
import os
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


def run_comparison(symbol):
    print(f"\nComparing results for {symbol}...")
    data = load_data(symbol)
    bar_count = len(data)
    initial_cash = 500000 if symbol == "BTCUSDT" else 5000

    # Load Original backtesting.py
    bt_orig = load_original_backtesting()
    orig_strat_cls = get_strategy_class(bt_orig.Strategy)

    print("Running Original backtesting.py...")
    start_time = time.time()
    bt = bt_orig.Backtest(
        data,
        orig_strat_cls,
        cash=initial_cash,
        commission=0.0008,
        exclusive_orders=True,
    )
    stats_orig = bt.run()
    orig_duration = time.time() - start_time

    # Run Tradelearn
    tl_strat_cls = get_strategy_class(tl_bt.Strategy)

    print("Running Tradelearn Facade...")
    start_time = time.time()
    bt_tl = tl_bt.Backtest(
        data,
        tl_strat_cls,
        cash=initial_cash,
        commission=0.0008,
        exclusive_orders=True,
    )
    stats_tl = bt_tl.run()
    tl_duration = time.time() - start_time

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

    for label, key in metrics:
        v_orig = stats_orig[key]
        v_tl = stats_tl[key]
        diff = v_tl - v_orig
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


if __name__ == "__main__":
    run_comparison("BTCUSDT")
    run_comparison("ETHUSDT")
