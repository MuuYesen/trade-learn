import time

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class SimpleStrategy(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.period)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()


def run_stress(n_bars=100_000, use_rust=True):
    # Generate fake data
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="1min")
    data = pd.DataFrame(
        {
            "open": np.random.randn(n_bars).cumsum() + 100,
            "high": np.random.randn(n_bars).cumsum() + 101,
            "low": np.random.randn(n_bars).cumsum() + 99,
            "close": np.random.randn(n_bars).cumsum() + 100,
            "volume": np.random.randint(100, 1000, n_bars),
        },
        index=dates,
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SimpleStrategy)
    cerebro.broker.setcash(100000.0)

    # Instantiate strategy and analyzers early to bypass run()'s auto-routing
    strategy = cerebro._instantiate_strategy()
    analyzers = cerebro._instantiate_analyzers(strategy)
    strategy.analyzers = analyzers
    strategy.start()

    start = time.perf_counter()
    if use_rust:
        cerebro._run_rust(strategy, analyzers)
    else:
        cerebro._run_python(strategy, analyzers)
    end = time.perf_counter()

    val = cerebro.broker.getvalue()
    elapsed = end - start
    mode = "Rust" if use_rust else "Python"
    print(f"[{mode}] N={n_bars} | Value: {val:.2f} | Time: {elapsed * 1000:.2f} ms")
    return elapsed


if __name__ == "__main__":
    print("Starting Stress Benchmark...")
    # Warmup
    run_stress(1000, use_rust=True)
    run_stress(1000, use_rust=False)

    n = 100_000
    t_rust = run_stress(n, use_rust=True)
    t_py = run_stress(n, use_rust=False)

    print(f"\nConclusion: Rust loop is {t_py / t_rust:.2f}x faster for N={n}")
