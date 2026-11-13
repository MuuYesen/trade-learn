"""C6 fund units are projections of the same backtest account, not another ledger."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def fund_profile(api,fund,mode,start,action=None):
    rows=[]
    frame=pd.DataFrame({n:[100.,110.,90.,120.,80.,140.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=6))
    class Business(api.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=1)
            if len(self)==4:self.sell(size=1)
            if len(self)==2 and action:
                if action=='setcash':self.broker.setcash(12000.)
                else:self.broker.add_cash(500. if action=='deposit' else -500.)
            rows.append((self.broker.fundmode,self.broker.fundvalue,self.broker.fundshares,self.broker.getcash(),self.broker.getvalue()))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000.)
    c.broker.set_fundmode(mode,fundstartval=start)
    c.adddata(api.feeds.PandasData(dataname=frame));c.addstrategy(Business)
    for name in ('Returns','SharpeRatio','DrawDown'):
        options=dict(fund=fund)
        if name!='DrawDown':options['timeframe']=api.TimeFrame.Days
        c.addanalyzer(getattr(api.analyzers,name),_name=name,**options)
    [strategy]=c.run()
    result={name:dict(strategy.analyzers.getbyname(name).get_analysis()) for name in ('Returns','SharpeRatio','DrawDown')}
    return rows,result


@pytest.mark.parametrize('fund',[None,False,True])
@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('start',[100.,250.])
def test_fund_units_and_three_analyzer_profiles_match_bt_without_cashflows(fund,mode,start):
    actual,analysis=fund_profile(tl,fund,mode,start)
    expected,golden=fund_profile(official,fund,mode,start)
    assert actual==expected
    for name in ('Returns','SharpeRatio'):assert analysis[name]==pytest.approx(golden[name])
    assert analysis['DrawDown']['max']==pytest.approx(dict(golden['DrawDown']['max']))
    assert {k:v for k,v in analysis['DrawDown'].items() if k!='max'}==pytest.approx({k:v for k,v in golden['DrawDown'].items() if k!='max'})


def test_engine_default_broker_cash_matches_fixed_bt():
    assert tl.Cerebro().broker.getcash()==official.Cerebro().broker.getcash()==10000.


@pytest.mark.parametrize('action',['setcash','deposit','withdrawal'])
@pytest.mark.parametrize('fund',[None,False,True])
@pytest.mark.parametrize('mode',[False,True])
def test_cash_mutation_three_analyzers_preserve_bt_fund_selection(action,fund,mode):
    actual,analysis=fund_profile(tl,fund,mode,100.,action)
    expected,golden=fund_profile(official,fund,mode,100.,action)
    for a,b in zip(actual,expected):assert a==pytest.approx(b,rel=1e-12,abs=1e-10)
    for name in ('Returns','SharpeRatio'):assert analysis[name]==pytest.approx(golden[name])
    assert analysis['DrawDown']['max']==pytest.approx(dict(golden['DrawDown']['max']))
    assert {k:v for k,v in analysis['DrawDown'].items() if k!='max'}==pytest.approx({k:v for k,v in golden['DrawDown'].items() if k!='max'})
