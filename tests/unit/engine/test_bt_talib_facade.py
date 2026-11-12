"""BT class/line facade delegates the qualified persistent numeric source."""
import inspect
import numpy as np
import pandas as pd
import backtrader as official
from tradelearn import engine as bt


def frame():
    close=np.linspace(10.,20.,40)+np.sin(np.arange(40))
    return pd.DataFrame({'open':close,'high':close+1,'low':close-1,'close':close,'volume':100.},index=pd.date_range('2026-01-01',periods=40))


def test_data_first_line_and_line_aliases_match_bt():
    data=bt.feeds.PandasData(dataname=frame());data._advance(2)
    assert data.lines.getlinealiases()==official.feeds.PandasData.lines.getlinealiases()
    assert data[0]==data.close[0] and data[-1]==data.close[-1]


def replay(api):
    class S(api.Strategy):
        def __init__(self):
            self.ma=api.talib.SMA(self.data.close,timeperiod=3)
            self.macd=api.talib.MACD(self.data.close,fastperiod=3,slowperiod=7,signalperiod=3)
            self.values=[]
        def prenext(self):self.next()
        def next(self):self.values.append((self.ma[0],self.macd.macd[0],self.macd.macdsignal[0],self.macd.macdhist[0]))
    c=api.Cerebro(stdstats=False);c.adddata(api.feeds.PandasData(dataname=frame()));c.addstrategy(S)
    return c.run()[0]


def test_bt_talib_is_indicator_class_and_preserves_official_output_aliases():
    assert inspect.isclass(bt.talib.SMA) and issubclass(bt.talib.SMA,bt.Indicator)
    expected=replay(official);actual=replay(bt)
    assert actual.ma.p.timeperiod==3
    assert actual.macd.lines.getlinealiases()==expected.macd.lines.getlinealiases()
    np.testing.assert_allclose(actual.values,expected.values,equal_nan=True,rtol=1e-10,atol=1e-10)
