import numpy as np
import pandas as pd
import pytest

import tradelearn.engine as bt
from tests.unit.backtest._engine_config import feed as _as_feed


class MultiTimeframeStrategy(bt.Strategy):
    def next(self):
        self.datas[0].close[0]
        d1_close = self.datas[1].close[0] if len(self.datas[1]) > 0 else None

        d0_ts = self.datas[0].datetime[0]
        d1_ts = self.datas[1].datetime[0] if len(self.datas[1]) > 0 else None

        print(f"[{self.datas[0].datetime.date(0)}] d0_ts: {d0_ts}, d1_ts: {d1_ts}, d1: {d1_close}")


def test_resampling_alignment():
    # 1-minute data for 15 minutes
    dates = pd.date_range("2023-01-01 10:00", periods=15, freq="1min")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 115, 15),
            "high": np.linspace(101, 116, 15),
            "low": np.linspace(99, 114, 15),
            "close": np.linspace(100.5, 115.5, 15),
            "volume": [100] * 15,
        },
        index=dates,
    )

    cerebro = bt.Cerebro()
    data0 = bt.feeds.PandasData(dataname=df, name="min1")
    cerebro.adddata(_as_feed(data0))

    # Use the new API
    with pytest.raises(NotImplementedError, match="incremental"):
        cerebro.resampledata(data0, timeframe=bt.TimeFrame.Minutes, compression=5)
    assert cerebro.datas == [data0]


if __name__ == "__main__":
    test_resampling_alignment()
