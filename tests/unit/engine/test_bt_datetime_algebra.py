"""BT date-line arithmetic must consume ordinal days rather than Unix seconds."""

import backtrader as official
import numpy as np
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime


def observations(api, stream=False):
    rows = []
    frame = pd.DataFrame(
        {name: [10.0] * 3 for name in ('open', 'high', 'low', 'close', 'volume')},
        index=pd.to_datetime(['2026-01-01', '2026-01-03', '2026-01-04']),
    )

    class Business(api.Strategy):
        def __init__(self):
            dates = self.data.datetime
            self.day_delta = dates - dates(-1)
            self.after_second = dates > api.date2num(pd.Timestamp('2026-01-02'))

        def next(self):
            rows.append((self.day_delta[0], self.after_second[0]))

    if stream:
        runtime = StrategyRuntime(Business, feeds={'A': 'bar'}, capacity=2)
        try:
            runtime.start()
            for stamp, bar in frame.iterrows():
                runtime.step_market('A', dict(ts=stamp.tz_localize('UTC'), **bar.to_dict()))
            runtime.stop()
        finally:
            runtime.cleanup()
    else:
        engine = api.Cerebro(stdstats=False)
        engine.adddata(api.feeds.PandasData(dataname=frame))
        engine.addstrategy(Business)
        engine.run()
    return rows


@pytest.mark.parametrize('stream', [False, True])
def test_data_datetime_lag_difference_and_comparison_match_actual_bt(stream):
    assert np.asarray(observations(tl, stream=stream)) == pytest.approx(
        np.asarray(observations(official)))
