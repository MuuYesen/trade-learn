"""Paired BT regressions derived from independent IA-C5/C6 frozen counterexamples."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl

def frame():
    prices=[10.,11.,12.,13.,14.,15.,16.,17.]
    return pd.DataFrame({**{k:prices for k in ('open','high','low','close')},
                         'volume':[1000.]*8},index=pd.date_range('2026-01-01',periods=8))

def trade_sequence(api, actions):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            action=actions.get(len(self))
            if action:
                getattr(self,action[0])(size=action[1])
        def notify_trade(self,t):
            rows.append({k:getattr(t,k,'MISSING') for k in
                         ('size','price','value','commission','pnl','pnlcomm','isopen','isclosed')})
    c=api.Cerebro(stdstats=False);c.broker.setcash(10000)
    c.broker.setcommission(commission=.001)
    c.adddata(api.feeds.PandasData(dataname=frame()),name='A');c.addstrategy(Business)
    [s]=c.run()
    return {'notices':rows,'position':[s.position.size,s.position.price],
            'cash':round(c.broker.getcash(),9)}

def observer_collection(api):
    rows=[]
    class Sample(api.Observer):
        lines=('sample',)
        def next(self):self.lines.sample[0]=len(self._owner)
    class Business(api.Strategy):
        def stop(self):
            for name,call in [('getitem',lambda:self.observers[0].sample[0]),
                              ('getnames',lambda:list(self.observers.getnames())),
                              ('getitems',lambda:[n for n,v in self.observers.getitems()])]:
                try:rows.append([name,call()])
                except Exception as e:rows.append([name,type(e).__name__])
    c=api.Cerebro(stdstats=False,runonce=False);c.addobserver(Sample)
    c.adddata(api.feeds.PandasData(dataname=frame()));c.addstrategy(Business);c.run()
    return rows

@pytest.mark.parametrize('actions',[
 {1:('buy',2),4:('sell',2)},
 {1:('buy',2),2:('buy',1),3:('sell',1),5:('sell',2)},
 {1:('buy',2),3:('sell',3),5:('buy',1)},
 {1:('sell',2),4:('buy',2)},
 {1:('sell',2),2:('sell',1),3:('buy',1),5:('buy',2)},
 {1:('sell',2),3:('buy',3),5:('sell',1)},
])
def test_trade_callbacks_cumulative_value_fees_partial_exit_and_split_reversal(actions):
    expected=trade_sequence(official,actions)
    actual=trade_sequence(tl,actions)
    assert actual['cash']==pytest.approx(expected['cash'])
    assert actual['position']==pytest.approx(expected['position'])
    assert len(actual['notices'])==len(expected['notices'])
    for ours,theirs in zip(actual['notices'],expected['notices'],strict=True):
        assert ours==pytest.approx(theirs)


def test_observer_collection_integer_index_names_and_items_match_fixed_bt():
    assert observer_collection(tl)==observer_collection(official)
