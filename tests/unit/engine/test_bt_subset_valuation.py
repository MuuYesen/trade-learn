"""P05 explicit BT valuation remains a view of the single native account."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def profile(api,alias,action):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            query=getattr(self.broker,alias)
            def record(label,value):
                rows.append((len(self),label,value,self.broker.getcash(),self.broker.getvalue(),self.broker.fundvalue,self.broker.fundshares))
            if len(self)==1:
                self.buy(data=self.datas[0],size=2.)
                self.sell(data=self.datas[1],size=.5)
                self.buy(data=self.datas[2],size=.25)
            if len(self)==3:
                if action=='deposit':self.broker.add_cash(500.)
                elif action=='withdraw':self.broker.add_cash(-250.)
                elif action=='setcash':self.broker.setcash(12000.)
            a,b,c=self.datas
            for label,datas in [('oneA',[a]),('oneB',[b]),('oneC',[c]),('two',[a,b]),('duplicate',[a,a]),('all',[]),('tuple',(a,b))]:
                record(label,query(datas))
            for mkt in (False,True):
                for lever in (False,True):record(f'global:{mkt}:{lever}',query(mkt=mkt,lever=lever))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000.);c.broker.setcommission(commission=.001)
    for i,values in enumerate([[10.,20.,30.,40.,50.],[100.,90.,80.,70.,60.],[20.,25.,30.,35.,40.]]):
        f=pd.DataFrame({n:values for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=5))
        c.adddata(api.feeds.PandasData(dataname=f),name=f'asset{i}')
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('alias',['getvalue','get_value'])
@pytest.mark.parametrize('action',['none','deposit','withdraw','setcash'])
def test_subset_single_raw_value_multi_cash_empty_all_duplicate_alias_and_native_cash_phase_match_bt(alias,action):
    actual=profile(tl,alias,action);expected=profile(official,alias,action)
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected):
        assert got[:2]==want[:2]
        assert got[2:]==pytest.approx(want[2:],rel=1e-12,abs=1e-9)


@pytest.mark.parametrize('foreign_kind',['foreign_feed','mixed','plain_object'])
def test_foreign_subset_is_rejected_before_settling_authorized_cash_or_native_account_mutation(foreign_kind):
    rejected=[];settled=[]
    f=pd.DataFrame({n:[10.,10.,10.] for n in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    foreign=tl.feeds.PandasData(dataname=f.copy())
    class Business(tl.Strategy):
        def next(self):
            if len(self)!=1:return
            self.broker.add_cash(500.)
            selected=([foreign] if foreign_kind=='foreign_feed' else [self.data,foreign] if foreign_kind=='mixed' else [object()])
            with pytest.raises(ValueError,match='does not belong'):self.broker.getvalue(selected)
            rejected.append((self.broker.getcash(),len(self.broker._pending_cash_requests)))
            settled.append((self.broker.getvalue([self.data]),self.broker.getcash(),len(self.broker._pending_cash_requests)))
    c=tl.Cerebro(stdstats=False);c.broker.setcash(10000.);c.adddata(tl.feeds.PandasData(dataname=f));c.addstrategy(Business);c.run()
    assert rejected==[(10000.,1)] and settled==[(0.,10500.,0)]


def direct_queries(api,alias,action):
    out=[]
    class Business(api.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=2.)
            if len(self)!=3:return
            query=getattr(self.broker,alias)
            def record(label):
                out.append((label,tuple(query(mkt=m,lever=l) for m in [False,True] for l in [False,True]),self.broker.getcash(),self.broker.fundvalue,self.broker.fundshares))
            record('before')
            if action=='set':self.broker.setcash(12000.)
            elif action=='add':self.broker.add_cash(500.)
            elif action=='remove':self.broker.add_cash(-250.)
            record('after_action')
            out.append(('single',query([self.data])))
            record('after_single')
            out.append(('empty',query([])))
            record('after_total')
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000);c.broker.setcommission(commission=.001)
    f=pd.DataFrame({n:[10.,20.,30.,40.,50.] for n in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=5))
    c.adddata(api.feeds.PandasData(dataname=f));c.addstrategy(Business);c.run();return out


@pytest.mark.parametrize('alias',['getvalue','get_value'])
@pytest.mark.parametrize('action',['none','set','add','remove'])
def test_standard_market_lever_cache_views_before_and_after_cash_commands_and_explicit_valuation(alias,action):
    got,want=direct_queries(tl,alias,action),direct_queries(official,alias,action)
    assert len(got)==len(want)
    for a,b in zip(got,want):
        assert a[0]==b[0]
        if len(a)==2:assert a[1]==pytest.approx(b[1])
        else:
            assert a[1]==pytest.approx(b[1]);assert a[2:]==pytest.approx(b[2:])
