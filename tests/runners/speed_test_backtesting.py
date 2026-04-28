import os
import sys
import time
from importlib import reload
from pathlib import Path

import pandas as pd

import tradelearn.compat.backtesting as tl_bt
from tradelearn.backtest.broker import RustBroker
from tradelearn.backtest.data import DataContainer
from tradelearn.backtest.engine import run_backtest
from tradelearn.backtest.strategy import Strategy as CoreStrategy


# 1. Load Original backtesting.py
def load_original_backtesting():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = sys.path[:]
    sys.path = [p for p in sys.path if p != current_dir and p != '']
    import backtesting as bt_orig
    reload(bt_orig)
    sys.path = orig_path
    return bt_orig

# 2. Strategy Definition for backtesting.py / Facade
def get_bt_strategy(BaseClass):
    class EMA_Cross_Strategy(BaseClass):
        ema_fast = 9
        ema_slow = 21
        def init(self):
            close = pd.Series(self.data.Close)
            self.ema9 = self.I(lambda: close.ewm(span=self.ema_fast, adjust=False).mean())
            self.ema21 = self.I(lambda: close.ewm(span=self.ema_slow, adjust=False).mean())
        def next(self):
            if len(self.ema9) < 2: return
            if not self.position and self.ema9[-2] <= self.ema21[-2] and self.ema9[-1] > self.ema21[-1]:
                self.buy()
            elif self.position and self.ema9[-2] >= self.ema21[-2] and self.ema9[-1] < self.ema21[-1]:
                self.position.close()
    return EMA_Cross_Strategy

class EMA_Cross_Core(CoreStrategy):
    params = dict(ema_fast=9, ema_slow=21)
    
    def init(self):
        # Vectorized indicator calculation using pandas
        close = pd.Series(self.data.get_array('close'))
        self.ema9 = close.ewm(span=self.params['ema_fast'], adjust=False).mean().values
        self.ema21 = close.ewm(span=self.params['ema_slow'], adjust=False).mean().values
        
    def next(self):
        cursor = self.data._cursor
        if cursor < 1: return
        # Direct array access via core attributes
        # In Core, self.ema9 is a Series/Array
        if not self.getposition().size and self.ema9[cursor-1] <= self.ema21[cursor-1] and self.ema9[cursor] > self.ema21[cursor]:
            self.buy()
        elif self.getposition().size and self.ema9[cursor-1] >= self.ema21[cursor-1] and self.ema9[cursor] < self.ema21[cursor]:
            self.close()

# 4. Data Loading
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "benchmarks" / "data" / "backtesting"


def load_data(symbol):
    filepath = DATA_DIR / f'{symbol}_30m.csv'
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df

def run_test():
    symbol = 'BTCUSDT'
    data = load_data(symbol)
    print(f"Data size: {len(data)} bars")
    
    # --- 1. Original backtesting.py ---
    bt_orig = load_original_backtesting()
    strat_orig = get_bt_strategy(bt_orig.Strategy)
    print("\nRunning Original backtesting.py...")
    start = time.time()
    bt = bt_orig.Backtest(data, strat_orig, cash=100000)
    res_orig = bt.run()
    t_orig = time.time() - start
    print(f"Time: {t_orig:.4f}s")

    # --- 2. Tradelearn Facade (Compatibility) ---
    strat_facade = get_bt_strategy(tl_bt.Strategy)
    print("\nRunning Tradelearn Facade (Compatibility)...")
    start = time.time()
    bt_tl = tl_bt.Backtest(data, strat_facade, cash=100000)
    res_facade = bt_tl.run()
    t_facade = time.time() - start
    print(f"Time: {t_facade:.4f}s")

    # --- 3. Tradelearn Core (High Performance) ---
    print("\nRunning Tradelearn Core (High Performance)...")
    # Mocking a cerebro-like setup for raw engine call
    class MockCerebro:
        def __init__(self, data, strategy_cls):
            self.datas = [data]
            self.strats = [(strategy_cls, (), {})]
            self.broker = RustBroker(cash=100000)
            self.match_mode = 'smart'
            from tradelearn.compat.backtrader.sizer import FixedSize
            self._sizer_spec = (FixedSize, {})
            self.analyzers = {}
    
    # Prepare data for core
    data_container = DataContainer(data)
    cerebro = MockCerebro(data_container, EMA_Cross_Core)
    
    start = time.time()
    res_core = run_backtest(cerebro)
    t_core = time.time() - start
    print(f"Time: {t_core:.4f}s")
    
    print("\n" + "="*40)
    print("SPEED COMPARISON")
    print("="*40)
    print(f"{'Engine':<25} | {'Time [s]':<10} | {'Relative'}")
    print("-" * 50)
    print(f"{'backtesting.py (Orig)':<25} | {t_orig:>10.4f} | 1.00x")
    print(f"{'Tradelearn (Facade)':<25} | {t_facade:>10.4f} | {t_facade/t_orig:>9.2f}x (Slower)")
    print(f"{'Tradelearn (Core)':<25} | {t_core:>10.4f} | {t_core/t_orig:>9.2f}x (FASTER!)")
    print("="*40)

if __name__ == '__main__':
    run_test()
