"""Fixed BT bar-level resample calls; no external provider or native execution."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def resample_calls(api,compression,start,keep_original):
    trace=[]
    frame=pd.DataFrame({'open':[10.,20.,30.,40.,50.,60.], 'high':[12.,22.,32.,42.,52.,62.],
        'low':[8.,18.,28.,38.,48.,58.], 'close':[11.,21.,31.,41.,51.,61.],
        'volume':[1.,2.,3.,4.,5.,6.], 'openinterest':[101.,102.,103.,104.,105.,106.]},
        index=pd.date_range(start,periods=6,freq='min'))
    feed=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes,compression=1)
    engine=api.Cerebro(stdstats=False,runonce=False)
    if keep_original:engine.adddata(feed,name='source')
    sampled=engine.resampledata(feed,name='sampled',timeframe=api.TimeFrame.Minutes,compression=compression)
    identity=sampled is feed
    metadata=(sampled._timeframe,sampled._compression,sampled.p.timeframe,sampled.p.compression)
    class Business(api.Strategy):
        def next(self):
            data=self.datas[-1]
            trace.append((len(data),data.datetime.datetime().isoformat(),data.open[0],data.high[0],data.low[0],data.close[0],data.volume[0],data.openinterest[0]))
    engine.addstrategy(Business);engine.run()
    return identity,metadata,trace


@pytest.mark.parametrize('compression',[2,3,5])
@pytest.mark.parametrize('start',['2026-01-01 09:00','2026-01-01 09:01'])
@pytest.mark.parametrize('keep_original',[False,True])
def test_bar_resample_edges_ohlcv_openinterest_identity_and_feed_metadata_match_bt(compression,start,keep_original):
    assert resample_calls(tl,compression,start,keep_original)==resample_calls(official,compression,start,keep_original)


def test_incremental_resample_does_not_emit_an_open_bucket_and_eof_is_once():
    from tradelearn.engine.resampling import _ClosedBarResampler
    builder=_ClosedBarResampler(120_000_000_000)
    row=dict(open=10.,high=12.,low=8.,close=11.,volume=1.,openinterest=101.)
    assert builder.push(60_000_000_000,row)==[]
    nextrow=dict(open=20.,high=22.,low=18.,close=21.,volume=2.,openinterest=102.)
    [(stamp,value,available)]=builder.push(120_000_000_000,nextrow)
    assert stamp==available==120_000_000_000
    assert value==dict(open=10.,high=22.,low=8.,close=21.,volume=3.,openinterest=102.)
    assert builder.push(180_000_000_000,row)==[]
    assert builder.finish()==[(240_000_000_000,row,240_000_000_000)]
    assert builder.finish()==[]
    with pytest.raises(RuntimeError,match='closed'):builder.push(300_000_000_000,row)


def resample_strategy(api,keep_original):
    trace=[]
    frame=pd.DataFrame({n:[10.,20.,30.,40.,50.,60.,70.,80.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01 09:00',periods=8,freq='min'))
    engine=api.Cerebro(stdstats=False,runonce=False)
    source=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes)
    if keep_original:engine.adddata(source,name='raw')
    sampled=engine.resampledata(source,name='sampled',timeframe=api.TimeFrame.Minutes,compression=2)
    class Business(api.Strategy):
        def __init__(self):
            if api is official:self.average=api.indicators.SMA(self.datas[-1].close,period=2)
            else:
                from tradelearn.indicators import talib
                self.average=talib.SMA(self.datas[-1].close,timeperiod=2)
            self.ordered=False
        def prenext(self):trace.append(('warmup',len(self),len(self.datas[-1])))
        def next(self):
            trace.append(('next',len(self),len(self.datas[-1]),self.average[0]))
            if not self.ordered:self.buy(data=self.datas[-1],size=2);self.ordered=True
        def notify_order(self,o):
            if o.status==api.Order.Completed:trace.append(('filled',o.executed.size,o.executed.price))
    engine.addstrategy(Business);engine.broker.setcash(10000)
    engine.run()
    return trace,engine.broker.getcash()


@pytest.mark.parametrize('keep_original',[False,True])
def test_resample_uses_shared_indicator_warmup_and_existing_order_cash_path(keep_original):
    actual,cash=resample_strategy(tl,keep_original)
    expected,expected_cash=resample_strategy(official,keep_original)
    assert actual==expected
    assert cash==expected_cash


def anchor_probe(api,seconds,compression):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            rows.append([self.data.datetime.datetime().isoformat(),self.data.open[0],
                         self.data.high[0],self.data.low[0],self.data.close[0],self.data.volume[0]])
    prices=[10.*i for i in range(1,11)]
    frame=pd.DataFrame({k:prices for k in ['open','high','low','close','volume']},
        index=pd.date_range('2026-01-02 09:00',periods=10,freq='s' if seconds else 'min'))
    timeframe=api.TimeFrame.Seconds if seconds else api.TimeFrame.Minutes
    c=api.Cerebro(stdstats=False,runonce=False)
    c.resampledata(api.feeds.PandasData(dataname=frame,timeframe=timeframe,compression=1),timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run();return rows

@pytest.mark.parametrize('seconds',[False,True])
@pytest.mark.parametrize('compression',[2,7,11])
def test_resample_nondivisor_compression_uses_bt_day_anchor(seconds,compression):
    assert anchor_probe(tl,seconds,compression)==anchor_probe(official,seconds,compression)


def delivery_probe(api,case):
    rows=[]
    class Business(api.Strategy):
        def prenext(self):self.next()
        def next(self):
            rows.append([self.datetime.datetime().isoformat(),
                         [[len(d),d.datetime.datetime().isoformat() if len(d) else None,
                           d.close[0] if len(d) else None] for d in self.datas]])
    if case=='sparse-clone':
        index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:09']);seconds=False;compression=2;clone=True
    else:
        seconds='seconds' in case;compression=2 if case.endswith('2') else 7;clone=False
        index=pd.date_range('2026-01-02 23:59:56' if seconds else '2026-01-02 23:56',periods=10,freq='s' if seconds else 'min')
    frame=pd.DataFrame({k:[(i+1)*10. for i in range(len(index))] for k in ['open','high','low','close','volume']},index=index)
    timeframe=api.TimeFrame.Seconds if seconds else api.TimeFrame.Minutes
    feed=api.feeds.PandasData(dataname=frame,timeframe=timeframe)
    c=api.Cerebro(stdstats=False,runonce=False)
    if clone:c.adddata(feed,name='original')
    c.resampledata(feed,name='sampled',timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run();return rows

@pytest.mark.parametrize('case',['sparse-clone','midnight-seconds2','midnight-seconds7','midnight-minutes7'])
def test_resample_preserves_source_delivery_availability_and_default_midnight_boundary(case):
    assert delivery_probe(tl,case)==delivery_probe(official,case)


def session_end_probe(api,clone,extra,order=False):
    rows=[];notices=[]
    index=pd.to_datetime(['2026-01-02 23:58','2026-01-02 23:59','2026-01-03 00:01']+(['2026-01-03 00:02','2026-01-03 00:03'] if extra else []))
    frame=pd.DataFrame({k:[(i+1)*10. for i in range(len(index))] for k in ['open','high','low','close','volume']},index=index)
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def prenext(self):self.next()
        def next(self):
            rows.append((len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.close[0] if len(d) else None) for d in self.datas)))
            if order and len(self.datas[-1]) and not self.sent:
                self.buy(data=self.datas[-1],size=1);self.sent=True
        def notify_order(self,o):
            if o.status==api.Order.Completed:notices.append((o.executed.dt,o.executed.price))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000.)
    feed=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(feed,name='raw')
    c.resampledata(feed,timeframe=api.TimeFrame.Minutes,compression=7)
    c.addstrategy(Business);c.run();return rows,notices,c.broker.getcash(),c.broker.getvalue()

@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('extra',[False,True])
def test_default_non_boundary_session_close_and_immediate_eof_match_bt(clone,extra):
    assert session_end_probe(tl,clone,extra)==session_end_probe(official,clone,extra)

@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('extra',[False,True])
def test_session_duplicate_labels_do_not_fill_market_before_bt_next_bar(clone,extra):
    assert session_end_probe(tl,clone,extra,True)==session_end_probe(official,clone,extra,True)


def test_native_execution_quote_shape_and_nonfinite_reject_before_mutation():
    from tradelearn import _rust
    engine=_rust.RustBacktestEngine([1],[30.],[50.],[30.],[50.],[100.],1000.,0.,False,False,False,0.,0.,False,False,False)
    step=engine.step_bars_with_quotes_compact
    bars=(['data0'],[1],[30.],[50.],[30.],[50.],[100.])
    with pytest.raises(ValueError,match='lengths differ'):step(*bars,[],[20.],[10.],[20.],False)
    with pytest.raises(ValueError,match='finite'):step(*bars,[float('nan')],[20.],[10.],[20.],False)
    assert engine.get_cash()==1000.
    assert engine.get_equity()==1000.
    assert engine.get_equity_curve()==([],[],[])
