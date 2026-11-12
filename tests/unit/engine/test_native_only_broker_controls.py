"""OHLCV simulation must not silently claim a venue-specific maker policy."""

import pandas as pd
import pytest

from tradelearn import engine as bt


@pytest.mark.parametrize("post_only,error", [(True, NotImplementedError), ("true", TypeError)])
def test_ohlcv_backend_does_not_ignore_explicit_native_post_only_policy(post_only, error):
    class Business(bt.Strategy):
        def next(self):
            self.buy(size=1, price=10, exectype=bt.Order.Limit, post_only=post_only)

    e = bt.Cerebro(stdstats=False)
    frame = pd.DataFrame(
        {key: [10.0] * 2 for key in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=2),
    )
    e.adddata(bt.feeds.PandasData(dataname=frame))
    e.addstrategy(Business)
    with pytest.raises(error, match="post_only|post-only"):
        e.run()
