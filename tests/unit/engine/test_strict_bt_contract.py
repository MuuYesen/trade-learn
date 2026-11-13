"""Behavior checked against installed frozen official Backtrader 1.9.78.123."""
import datetime
import inspect
import pandas as pd
import pytest
import backtrader as official
import tradelearn.engine as bt


def _minimum(strategy, period):
    class Minimum(bt.Indicator):
        lines=('sample',)
        def __init__(self):self.addminperiod(period)
        def next(self):self.lines.sample[0]=self.data.close[0]
    return Minimum(strategy.data)


def bars():
    return pd.DataFrame(dict(open=[10.,11.,12.],high=[11.,12.,13.],low=[9.,10.,11.],close=[10.,11.,12.],volume=[100.]*3),index=pd.date_range('2026-01-01',periods=3))


def test_date_numbers_are_bt_ordinals_and_timezone_behavior():
    dt=datetime.datetime(2026,1,2,3,4,5,123456,tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
    assert bt.date2num(dt)==official.date2num(dt)
    for naive in [True,False]:
        assert bt.num2date(bt.date2num(dt),tz=datetime.timezone.utc,naive=naive)==official.num2date(official.date2num(dt),tz=datetime.timezone.utc,naive=naive)


def test_timeframe_matches_fixed_bt_values_and_names():
    for name in ['Ticks','MicroSeconds','Seconds','Minutes','Days','Weeks','Months','Years','NoTimeFrame']:
        assert getattr(bt.TimeFrame,name)==getattr(official.TimeFrame,name)
        for compression in [1,5]:
            assert bt.TimeFrame.getname(getattr(bt.TimeFrame,name),compression)==official.TimeFrame.getname(getattr(official.TimeFrame,name),compression)


def test_addstrategy_returns_registration_index():
    c=bt.Cerebro()
    assert c.addstrategy(bt.Strategy)==0
    assert c.addstrategy(bt.Strategy)==1


@pytest.mark.parametrize('method,args',[('replaydata',(None,)),('chaindata',(None,)),('rolloverdata',(None,))])
def test_unimplemented_bt_semantics_rejected_before_mutation(method,args):
    c=bt.Cerebro()
    with pytest.raises(NotImplementedError): getattr(c,method)(*args)
    assert c.datas==[]


def test_nextstart_runs_once_after_warmup():
    class S(bt.Strategy):
        def __init__(self): self.calls=[];self.minimum=_minimum(self,2)
        def prenext(self): self.calls.append(('prenext',len(self)))
        def nextstart(self): self.calls.append(('nextstart',len(self)))
        def next(self): self.calls.append(('next',len(self)))
    c=bt.Cerebro();c.adddata(bt.feeds.PandasData(dataname=bars()));c.addstrategy(S)
    assert c.run()[0].calls==[('prenext',1),('nextstart',2),('next',3)]


def test_buy_sell_signatures_match_bt():
    for name in ['buy','sell']:
        ours=inspect.signature(getattr(bt.Strategy,name)); theirs=inspect.signature(getattr(official.Strategy,name))
        assert list(ours.parameters)==list(theirs.parameters)
        assert [p.default for p in ours.parameters.values()]==[p.default for p in theirs.parameters.values()]


def test_order_enumerations_match_official():
    for name in ['Buy','Sell','Market','Close','Limit','Stop','StopLimit','StopTrail','StopTrailLimit']:
        assert getattr(bt.Order,name)==getattr(official.Order,name)

@pytest.mark.parametrize('kw',[{},dict(commission=.1),dict(commission=2,margin=100,mult=10),dict(commission=.002,percabs=True,mult=10),dict(margin=2,automargin=-1,mult=10)])
def test_commission_semantics_match_official(kw):
    ours=bt.CommInfoBase(**kw);theirs=official.CommInfoBase(**kw)
    for method,args in [('getcommission',(2,100)),('getoperationcost',(-2,100)),('getvaluesize',(-2,100)),('getsize',(100,1000)),('profitandloss',(2,100,105))]:
        assert getattr(ours,method)(*args)==getattr(theirs,method)(*args)

@pytest.mark.parametrize('options',[{'typo':1},{'mode':'live'},{'trade_on_close':True},{'exactbars':True},{'cheat_on_open':True}])
def test_cerebro_rejects_non_bt_or_unimplemented_configuration(options):
    with pytest.raises((TypeError,NotImplementedError)):bt.Cerebro(**options)


def test_opt_results_are_nested_and_default_lightweight():
    class S(bt.Strategy):
        params=(('value',1),)
    c=bt.Cerebro();c.adddata(bt.feeds.PandasData(dataname=bars()));c.optstrategy(S,value=[1,2])
    result=c.run()
    assert len(result)==2 and all(isinstance(item,list) for item in result)
    assert [item[0].p.value for item in result]==[1,2]
    assert all(isinstance(item[0],bt.OptReturn) for item in result)


def test_engine_zero_size_is_no_order_and_negative_size_is_absolute():
    class S(bt.Strategy):
        def next(self):
            if len(self)==1:
                assert self.buy(size=0) is None
                self.order=self.buy(size=-2)
    c=bt.Cerebro();c.adddata(bt.feeds.PandasData(dataname=bars()));c.addstrategy(S)
    assert c.run()[0].order.size==2


def test_pandasdata_autodetects_case_columns_and_missing_optional_lines():
    frame=bars().rename(columns=str.upper).drop(columns='VOLUME')
    data=bt.feeds.PandasData(dataname=frame,timeframe=bt.TimeFrame.Days,compression=1)
    data._advance(0)
    assert data.close[0]==10
    assert pd.isna(data.volume[0])
    assert pd.isna(data.openinterest[0])


def test_pandasdata_positional_columns_datetime_and_range():
    frame=bars().reset_index().rename(columns={'index':'when'})
    data=bt.feeds.PandasData(dataname=frame,datetime=0,open=1,high=2,low=3,close=4,volume=5,
                            fromdate=pd.Timestamp('2026-01-02'),todate=pd.Timestamp('2026-01-03'))
    assert data.buflen()==2
    data._advance(0)
    assert data.close[0]==11


def test_analyzer_bt_callbacks_and_stop_analysis_are_real():
    class A(bt.Analyzer):
        def __init__(self):self.calls=[]
        def prenext(self):self.calls.append('pre')
        def nextstart(self):self.calls.append('first')
        def next(self):self.calls.append('next')
        def notify_order(self,o):self.calls.append(o.getstatusname())
        def stop(self):self.rets={'calls':list(self.calls)}
    class S(bt.Strategy):
        def __init__(self):self.minimum=_minimum(self,2)
        def next(self):
            if len(self)==2:self.buy(size=1)
    c=bt.Cerebro();c.adddata(bt.feeds.PandasData(dataname=bars()));c.addstrategy(S);c.addanalyzer(A,_name='a')
    s=c.run()[0]
    assert s.analyzers.a.get_analysis()['calls']==['pre','first','Submitted','Accepted','Completed','next']
    assert s.analyzer_results['a']==s.analyzers.a.get_analysis()


def test_position_update_return_and_fractional_truth_match_bt():
    ours=bt.Position();theirs=official.Position()
    for size,price in [(2,10),(-1,11),(-3,12),(2,13),(1e-10,14)]:
        assert ours.update(size,price)==theirs.update(size,price)
        assert bool(ours)==bool(theirs)
    assert not callable(ours)


def test_position_queries_accept_explicit_broker_and_default_name():
    class S(bt.Strategy):
        def next(self):
            assert self.getposition(broker=self.broker) is self.getpositionbyname()
            assert self.getpositionsbyname()['A'] is self.position
    c=bt.Cerebro();c.adddata(bt.feeds.PandasData(dataname=bars()),name='A');c.addstrategy(S);c.run()


def test_adddata_rejects_non_bt_dataframe_convenience():
    with pytest.raises(TypeError):bt.Cerebro().adddata(bars())


def test_pandasdata_declared_extra_lines_keep_inherited_line_order():
    class Extra(bt.feeds.PandasData):
        lines=('factor',)
        params=(('factor',-1),)
    feed=Extra(dataname=bars().assign(FACTOR=[1.,2.,3.]))
    feed._advance(1)
    assert feed.factor[0]==2
    assert feed.lines[0] is feed.close
    assert feed.lines[1] is feed.low


def test_order_queries_and_safe_open_order_snapshot_match_bt():
    from tradelearn.backtest.models import Order
    from tradelearn.backtest.broker import RustBroker
    order=Order(ref=1,data=object(),ordtype=Order.Buy,size=2)
    assert order.alive() is True
    assert order.getstatusname(Order.Partial)=='Partial'
    assert order.getordername(Order.StopLimit)=='StopLimit'
    order.status=Order.Partial;order.executed.size=1
    broker=RustBroker(cash=1000);broker._active_orders[1]=order
    snapshot=broker.get_orders_open(safe=True)[0]
    assert snapshot is not order and snapshot.executed is not order.executed
    snapshot.status=Order.Canceled;snapshot.executed.size=2
    assert order.status==Order.Partial and order.executed.size==1


def test_qualified_microsecond_resample_registers_the_feed():
    from tradelearn.engine import Cerebro,feeds,TimeFrame
    engine=Cerebro();data=feeds.PandasData(dataname=pd.DataFrame({'close':[1.,2.]},index=pd.date_range('2026-01-01',periods=2)))
    sampled=engine.resampledata(data,timeframe=TimeFrame.MicroSeconds)
    assert engine.datas==[sampled]
    assert sampled._timeframe==TimeFrame.MicroSeconds
    assert list(sampled._frame.close)==[1.,2.]


@pytest.mark.parametrize('on_close',[False,True])
def test_terminal_submission_retains_pending_order_without_fabricated_notifications(on_close):
    from tradelearn.engine import Cerebro,Strategy,feeds,Order
    class S(Strategy):
        def __init__(self):self.events=[]
        def next(self):
            if len(self)==2:self.order=self.buy(size=1)
        def notify_order(self,order):self.events.append(order.status)
    frame=pd.DataFrame({k:[1.,2.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=2))
    engine=Cerebro(stdstats=False);engine.broker.set_coc(on_close)
    engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S);s=engine.run()[0]
    assert s.order.status==Order.Submitted and s.events==[]
    assert len(s.stats.orders)==1 and s.stats.fills.empty
    assert engine.broker.get_orders_open()==[]


def test_default_broker_target_value_uses_commission_sizing_and_close_zero_is_noop():
    from tradelearn.engine import Cerebro,Strategy,feeds
    class S(Strategy):
        def next(self):
            if len(self)==1:self.order_target_value(target=20.)
            elif len(self)==2:self.zero=self.close(size=0)
    frame=pd.DataFrame({k:[10.,10.,10.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine=Cerebro(stdstats=False);engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.position.size==2 and s.zero is None
    assert len(s.stats.fills)==1


def test_order_notifications_follow_next_broker_cycle_and_snapshot_status():
    from tradelearn.engine import Cerebro,Strategy,feeds,Order
    class S(Strategy):
        def __init__(self):self.notifications=[]
        def next(self):
            if len(self)==1:self.order=self.buy(size=1)
        def notify_order(self,order):
            self.notifications.append((len(self),order.status,order is self.order))
    frame=pd.DataFrame({k:[10.,11.,12.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine=Cerebro(stdstats=False);engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.notifications==[(2,Order.Submitted,False),(2,Order.Accepted,False),(2,Order.Completed,False)]
    assert s.order.status==Order.Completed


def test_sell_execution_quantity_is_signed_like_bt():
    from tradelearn.engine import Cerebro,Strategy,feeds
    class S(Strategy):
        def next(self):
            if len(self)==1:self.order=self.sell(size=2)
    frame=pd.DataFrame({k:[10.,11.,12.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine=Cerebro(stdstats=False);engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.order.executed.size==-2


def test_setbroker_returns_broker_and_binds_cerebro():
    from tradelearn.engine import Cerebro
    from tradelearn.backtest.broker import RustBroker
    engine=Cerebro();broker=RustBroker(cash=1000)
    assert engine.setbroker(broker) is broker
    assert engine.getbroker() is broker and broker.cerebro is engine


def test_writer_lifecycle_is_rejected_before_registering_a_stub():
    from tradelearn.engine import Cerebro
    engine=Cerebro()
    with pytest.raises(NotImplementedError,match='writer'):
        engine.addwriter(object)
    assert engine.writers==[]


def test_public_order_reference_stays_stable_across_rust_admission_and_fill():
    from tradelearn.engine import Cerebro,Strategy,feeds
    class S(Strategy):
        def __init__(self):self.refs=[]
        def next(self):
            if len(self)==1:
                self.order=self.buy(size=1)
                self.submitted_ref=self.order.ref
        def notify_order(self,order):self.refs.append(order.ref)
    frame=pd.DataFrame({k:[10.,11.,12.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine=Cerebro(stdstats=False);engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.order.ref==s.submitted_ref
    assert s.refs==[s.submitted_ref]*3
    assert engine.broker._active_orders=={}


def test_broker_commission_parameters_and_percent_normalization_are_executed():
    import inspect
    import backtrader as official
    from tradelearn.engine import Cerebro,Strategy,feeds
    engine=Cerebro(stdstats=False)
    actual=inspect.signature(type(engine.broker).setcommission)
    expected=inspect.signature(official.brokers.BackBroker.setcommission)
    assert [(p.name,p.default) for p in actual.parameters.values()]==[(p.name,p.default) for p in expected.parameters.values()]
    engine.broker.setcash(10000)
    engine.broker.setcommission(commission=.1,percabs=False)
    info=engine.broker.getcommissioninfo(None)
    assert info.getcommission(2,100)==pytest.approx(.2)
    class S(Strategy):
        def next(self):
            if len(self)==1:self.buy(size=2)
    frame=pd.DataFrame({k:[100.]*3 for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.stats.fills.commission.tolist()==pytest.approx([.2])
    assert s.stats.summary['final_value']==pytest.approx(9999.8)


@pytest.mark.parametrize('options',[{'margin':10},{'mult':5},{'commtype':1},{'leverage':2},{'interest':.05},{'name':'A'}])
def test_unqualified_broker_economics_reject_before_mutation(options):
    from tradelearn.engine import Cerebro
    broker=Cerebro().broker;initial=broker.getcommissioninfo(None)
    with pytest.raises(NotImplementedError,match='commission|economics'):
        broker.setcommission(**options)
    assert broker.getcommissioninfo(None) is initial


def test_python_commission_override_is_not_silently_ignored_by_native_execution():
    from tradelearn.engine import Cerebro,CommInfoBase
    class Custom(CommInfoBase):
        def getcommission(self,size,price):return 10.
    broker=Cerebro().broker
    with pytest.raises(NotImplementedError,match='commission'):
        broker.addcommissioninfo(Custom())


def test_order_requested_size_is_signed_while_submission_quantity_remains_positive():
    from tradelearn.engine import Cerebro,Strategy,feeds
    from tradelearn.core.strategy_events import ObserverSpec
    intents=[]
    def factory(strategy):
        return ObserverSpec(observer=lambda event:intents.append(event.details['size']),
                            events=frozenset({'intent.submit_order'}),phases=('before',))
    class S(Strategy):
        strategy_observer_factories=(factory,)
        def next(self):
            if len(self)==1:self.order=self.sell(size=2)
    frame=pd.DataFrame({k:[10.,11.,12.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    engine=Cerebro(stdstats=False);engine.adddata(feeds.PandasData(dataname=frame));engine.addstrategy(S)
    s=engine.run()[0]
    assert s.order.size==-2 and s.order.executed.size==-2
    assert intents==[2.]
    assert s.position.size==-2


def test_unqualified_bt_plot_does_not_return_a_different_report_chart():
    from tradelearn.engine import Cerebro
    with pytest.raises(NotImplementedError,match='plot'):
        Cerebro().plot()


def test_order_creation_and_execution_dates_follow_bt_without_changing_result_clock():
    from tradelearn import engine as bt
    class S(bt.Strategy):
        def next(self):
            if len(self)==1:self.order=self.sell(size=2)
    frame=pd.DataFrame({k:[10.,11.,12.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    c=bt.Cerebro(stdstats=False);c.adddata(bt.feeds.PandasData(dataname=frame));c.addstrategy(S)
    s=c.run()[0];o=s.order
    assert o.created.dt==bt.date2num(frame.index[0])
    assert (o.created.size,o.created.price,o.created.pricelimit)==(-2,10.,10.)
    assert o.executed.dt==bt.date2num(frame.index[1]) and o.executed.remsize==0
    assert s.stats.orders.executed_dt.iloc[0]==pd.Timestamp('2026-01-02',tz='UTC')


def test_sizer_signed_quantity_is_normalized_like_official_bt():
    results = []
    for api in (official, bt):
        class SignedSizer(api.Sizer):
            def _getsizing(self, comminfo, cash, data, isbuy):
                return -2

        class S(api.Strategy):
            def next(self):
                if len(self) == 1:
                    self.submitted = self.buy()

        c = api.Cerebro()
        c.adddata(api.feeds.PandasData(dataname=bars()))
        c.addsizer(SignedSizer)
        c.addstrategy(S)
        strategy = c.run()[0]
        results.append((strategy.submitted.size, strategy.submitted.isbuy()))
    assert results == [(2, True), (2, True)]


def test_add_timer_registration_and_signature_match_fixed_bt():
    ours=inspect.signature(bt.Cerebro.add_timer);theirs=inspect.signature(official.Cerebro.add_timer)
    assert list(ours.parameters)==list(theirs.parameters)
    assert [p.default for p in ours.parameters.values()]==[p.default for p in theirs.parameters.values()]
    c=bt.Cerebro();timer=c.add_timer(datetime.time(9),strats=True,marker=42)
    assert timer.p.tid==0 and timer.p.owner is c and timer.p.strats is True
    assert timer.kwargs=={'marker':42} and c.datas==[]
