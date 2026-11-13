"""Explicit feed identity and an empty visible first-bar clone are legal BT states."""
import math
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def profile(api,kind,features,unique,takelate,adjusted=False):
    times={'duplicates':['09:00','09:01','09:01','09:08','09:09','09:16'],
           'late-open':['09:00','09:04','09:02','09:08','09:09','09:16'],
           'late_closed':['09:00','09:08','09:01','09:09','09:16','09:17'],
           'late_eof_back':['09:00','09:08','09:16','09:02','09:01','09:09'],
           'ordered':['09:00','09:01','09:08','09:09','09:16','09:17']}[kind]
    values=[100.,107.,94.,115.,81.,120.]
    frame=pd.DataFrame({n:values for n in ['open','high','low','close','volume']},index=pd.to_datetime(['2026-01-02 '+t for t in times]))
    rows=[];calls=[]
    class Counter(api.Indicator):
        lines=('value',)
        def __init__(self):self.count=0
        def next(self):
            self.count+=1;self.value[0]=self.count;calls.append((len(self),len(self.data),self.count))
    def number(x):return 'NaN' if isinstance(x,float) and math.isnan(x) else x
    class Business(api.Strategy):
        def __init__(self):
            if features!='plain':self.counter=Counter(self.data0);self.average=api.talib.SMA(self.data0.close,timeperiod=2)
            if features=='both':self.other=Counter(self.datas[-1])
        def record(self,phase):
            row=(phase,len(self),tuple((len(d),d.datetime[0],d.close[0]) if len(d) else (0,None) for d in self.datas))
            if features!='plain':row+=(len(self.counter),number(self.counter[0]),number(self.average[0]))
            rows.append(row)
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    c=api.Cerebro(stdstats=False,runonce=False);raw=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes)
    c.adddata(raw,name='raw');c.replaydata(raw,timeframe=api.TimeFrame.Minutes,compression=7,takelate=takelate,adjbartime=adjusted,**({'name':'sample'} if unique else {}))
    c.addstrategy(Business);c.run();return rows,calls


@pytest.mark.parametrize('kind',['duplicates','late-open','ordered','late_closed','late_eof_back'])
@pytest.mark.parametrize('features',['plain','raw','both'])
@pytest.mark.parametrize('unique',[False,True])
@pytest.mark.parametrize('takelate',[False,True])
@pytest.mark.parametrize('adjusted',[False,True])
def test_first_logical_bar_withdrawal_and_duplicate_public_names_preserve_explicit_indicator_owner(kind,features,unique,takelate,adjusted):
    assert profile(tl,kind,features,unique,takelate,adjusted)==profile(official,kind,features,unique,takelate,adjusted)
