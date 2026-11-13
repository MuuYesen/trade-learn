"""P05 mutations must reach the one native account, including callbacks."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def cash_profile(api,action,trading,fundmode,dispatch="optimized"):
    rows=[]
    class Business(api.Strategy):
        def record(self,label):
            rows.append((label,len(self),self.broker.getcash(),self.broker.getvalue(),self.broker.fundvalue,self.broker.fundshares))
        def next(self):
            self.record('before')
            if len(self)==1 and trading:self.buy(size=1.)
            if len(self)==2:
                if action=='setcash':result=self.broker.setcash(12000.)
                elif action=='deposit':result=self.broker.add_cash(500.)
                else:result=self.broker.add_cash(-500.)
                assert result is None
                self.record('after')
            if len(self)==4 and trading:self.sell(size=1.)
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000.)
    c.broker.set_fundmode(fundmode)
    f=pd.DataFrame({n:[10.,20.,30.,40.,50.,60.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=6))
    source=api.feeds.PandasData(dataname=f)
    c.adddata(source)
    if dispatch=='coordinator':
        from datetime import time
        c.add_timer(time(23,59),cheat=True)
    elif dispatch=='multi':c.adddata(api.feeds.PandasData(dataname=f.copy()))
    elif dispatch=='replay':c.replaydata(source,timeframe=api.TimeFrame.Days,compression=2)
    c.addstrategy(Business);c.run()
    return rows


@pytest.mark.parametrize('action',['setcash','deposit','withdrawal'])
@pytest.mark.parametrize('trading',[False,True])
@pytest.mark.parametrize('fundmode',[False,True])
def test_runtime_cash_mutation_and_fund_units_match_fixed_bt_without_second_account(action,trading,fundmode):
    actual=cash_profile(tl,action,trading,fundmode);expected=cash_profile(official,action,trading,fundmode)
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected):
        assert got[:2]==want[:2]
        assert got[2:]==pytest.approx(want[2:],rel=1e-12,abs=1e-10)


def matching_boundary(api,action):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            rows.append(('next',len(self),self.broker.getcash(),self.broker.getvalue(),self.broker.fundshares))
            if len(self)==1:
                if action=='setcash':self.broker.setcash(600.)
                else:self.broker.add_cash(500. if action=='deposit' else -50.)
                self.buy(size=20.)
        def notify_order(self,o):rows.append(('order',o.getstatusname(),o.executed.size,self.broker.getcash(),len(self)))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(100.)
    f=pd.DataFrame({n:[10.]*4 for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=4))
    c.adddata(api.feeds.PandasData(dataname=f));c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('action',['setcash','deposit','withdrawal'])
def test_added_cash_does_not_fund_the_match_before_fixed_bt_valuation_phase(action):
    assert matching_boundary(tl,action)==matching_boundary(official,action)


@pytest.mark.parametrize('dispatch',['coordinator','multi','replay'])
@pytest.mark.parametrize('action',['setcash','deposit','withdrawal'])
def test_all_existing_runners_apply_one_native_cash_flow_and_fund_units(dispatch,action):
    actual=cash_profile(tl,action,True,True,dispatch)
    expected=cash_profile(official,action,True,True,dispatch)
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected):
        assert got[:2]==want[:2]
        assert got[2:]==pytest.approx(want[2:],rel=1e-12,abs=1e-10)
