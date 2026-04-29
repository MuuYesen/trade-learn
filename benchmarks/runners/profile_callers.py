import cProfile
import io
import pstats

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class SimpleStrategy(bt.Strategy):
    def next(self):
        _ = self.data.close[0]


def profile_callers(n_bars=5000):
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
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
    ps.print_stats(20)

    print("\nCALLERS OF _box_func (Pandas Datetime conversion):")
    ps.print_callers("_box_func")

    print(s.getvalue())


if __name__ == "__main__":
    profile_callers(5000)
