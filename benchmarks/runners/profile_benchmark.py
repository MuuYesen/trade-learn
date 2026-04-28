import cProfile
import io
import pstats

import numpy as np
import pandas as pd

import tradelearn.compat.backtrader as bt


class SimpleStrategy(bt.Strategy):
    def next(self):
        # Access some data to simulate real work
        c = self.data.close[0]
        if not self.position:
            if c > 100:
                self.buy()
        else:
            if c < 99:
                self.sell()


def profile_tradelearn(n_bars=50_000):
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

    pr = cProfile.Profile()
    pr.enable()
    cerebro.run()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    print("Profiling Tradelearn with Rust Loop (N=50,000)...")
    profile_tradelearn(50000)
