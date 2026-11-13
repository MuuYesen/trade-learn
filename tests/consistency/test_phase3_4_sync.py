"""The supported resample contract follows the fixed BT completion clock."""
import backtrader as official
import numpy as np
import pandas as pd

import tradelearn.engine as bt


def test_resampling_alignment():
    dates = pd.date_range("2023-01-01 10:00", periods=15, freq="1min")
    frame = pd.DataFrame(
        {
            "open": np.linspace(100, 115, 15),
            "high": np.linspace(101, 116, 15),
            "low": np.linspace(99, 114, 15),
            "close": np.linspace(100.5, 115.5, 15),
            "volume": [100] * 15,
        },
        index=dates,
    )
    def business(api, observed):
        class Business(api.Strategy):
            def next(self):
                sample = self.datas[1]
                observed.append((len(self.data), len(sample), sample.datetime[0],
                                 *[float(getattr(sample, field)[0]) for field in
                                   ("open", "high", "low", "close", "volume")]))

        return Business

    results = []
    for api in (official, bt):
        observed = []

        cerebro = api.Cerebro(stdstats=False, runonce=False)
        raw = api.feeds.PandasData(dataname=frame.copy(), timeframe=api.TimeFrame.Minutes)
        cerebro.adddata(raw, name="min1")
        sample = cerebro.resampledata(raw, timeframe=api.TimeFrame.Minutes,
                                     compression=5, name="min5")
        assert sample is not raw and len(cerebro.datas) == 2
        cerebro.addstrategy(business(api, observed))
        cerebro.run()
        results.append(observed)
    assert results[1] == results[0]
    assert len(results[0]) > 0
