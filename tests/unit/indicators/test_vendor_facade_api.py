"""Engine/Lite use the same indicator objects through explicit source imports."""

import numpy as np
import pandas as pd
import pytest

import tradelearn as tl
import tradelearn.engine as bt
import tradelearn.lite as lite
from tradelearn.indicators import talib, tdx


@pytest.fixture
def tv():
    import sys

    if sys.version_info < (3, 11):
        pytest.skip("Fixed PyneCore requires Python 3.11+; full vendor composition unavailable")
    from tradelearn.indicators import tv

    return tv


def bars():
    x = np.arange(50, dtype=float) + 10
    return pd.DataFrame(
        dict(open=x, high=x + 1, low=x - 1, close=x, volume=x * 10),
        index=pd.date_range("2026-01-01", periods=len(x), tz="UTC"),
    )


def test_root_shares_explicit_source_modules_and_retires_pta(tv):
    assert tl.talib is talib and tl.tdx is tdx and tl.tv is tv
    assert not hasattr(tl, "pta")
    for name in ["pta", "tdx", "tv"]:
        assert not hasattr(bt, name)


@pytest.mark.parametrize("facade", [bt, lite])
def test_actual_batch_facades_use_qualified_source_calls(facade, tv):
    class S(facade.Strategy):
        def init(self):
            self.ma = talib.SMA(self.data.close, timeperiod=2)
            self.tdx = tdx.MA(self.data.close, N=2)
            self.tv = tv.sma(self.data.close, length=2)
            self.seen = []

        def next(self):
            self.seen.append((self.ma[0], self.tdx[0], self.tv[0]))

    if facade is bt:
        S.__init__ = S.init
        c = bt.Cerebro()
        c.adddata(bt.feeds.PandasData(dataname=bars()))
        c.addstrategy(S)
        strategy = c.run()[0]
    else:
        strategy = lite.Backtest(bars(), S).run().strategy
    assert strategy.seen
    np.testing.assert_allclose(strategy.seen[-1], [58.5] * 3)


def test_case_aliases_and_source_implementation_boundaries(tv):
    assert talib.SMA is talib.sma
    assert tv.SMA is tv.sma
    pd.testing.assert_series_equal(tdx.MA(bars().close, N=3), tdx.ma(bars().close, n=3))
    assert talib.SMA._func.__module__ == "tradelearn.indicators.talib.persistent"
    assert tv.SMA._func.__module__ == "tradelearn.indicators.tv.persistent"
    assert tdx.MA._func.__module__ == "tradelearn.indicators.tdx.upstream"
