import numpy as np
import pandas as pd
import pytest
from tradelearn.backtest.lines import Lines, LineSeries
from tradelearn.engine.datafeed import DataFeed


def test_lines_register_distinct_equal_valued_lines_by_identity():
    owner=type('Owner',(),{})()
    lines=Lines(owner)
    lines.a=LineSeries([1,2]); lines.b=LineSeries([1,2])
    assert len(lines)==2
    assert lines[0] is lines.a and lines[1] is lines.b


def test_cursor_source_applies_to_all_offsets_and_length():
    feed=DataFeed(pd.DataFrame(dict(open=[1,2,3,4],high=[1,2,3,4],low=[1,2,3,4],close=[1,2,3,4]),index=pd.date_range('2026-01-01',periods=4)))
    line=feed.close*2
    feed._advance(3)
    assert line[-2]==4
    assert len(line)==4


def test_line_window_snapshot_cannot_mutate_feed():
    line=LineSeries([1,2,3]);line._advance(2)
    result=line.get(size=2);result[:]=99
    assert line[0]==3


def test_line_math_supports_reverse_ops():
    line=LineSeries([1,2,4]);line._advance(2)
    assert (10-line)[0]==6
    assert (2+line)[0]==6
    assert (8/line)[0]==2


def test_public_conversions_never_reveal_future_during_init_or_next():
    from tradelearn.engine import Cerebro,Strategy,feeds
    frame=pd.DataFrame({k:[1.,2.,3.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    class S(Strategy):
        def __init__(self):
            assert self.data.close.to_series().empty
            assert len(self.data.get_array('close'))==0
            with pytest.raises(ValueError): self.data.close[1]
        def next(self):
            assert len(self.data.close.to_series())==len(self)
            assert len(self.data.get_array('close'))==len(self)
    c=Cerebro();c.adddata(feeds.PandasData(dataname=frame));c.addstrategy(S);c.run()
