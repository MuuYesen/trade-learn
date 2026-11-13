"""P04 timer registration belongs to the BT owner and startup lifecycle."""
from datetime import time,timedelta
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


def profile(api,owner,phase,cheat,strats):
    events=[];handles=[];instances=[]
    def received(who,timer,when,**kwargs):
        events.append((who,timer.p.tid,timer.p.owner is (instances[0] if owner=='strategy' else engine),
                       when.isoformat(),len(instances[0]),kwargs))
    class App(api.Cerebro):
        def notify_timer(self,timer,when,**kwargs):received('cerebro',timer,when,**kwargs)
    class Business(api.Strategy):
        def __init__(self):
            instances.append(self)
            if phase=='init':self.register()
        def register(self):
            selected=self if owner=='strategy' else self.cerebro
            options=dict(cheat=cheat,repeat=timedelta(minutes=1),token='user-timer')
            if owner=='cerebro':options['strats']=strats
            timer=selected.add_timer(time(9,1),**options)
            handles.append((timer.p.tid,timer.p.owner is selected,timer.p.when,timer.p.repeat,timer.kwargs))
        def start(self):
            if phase=='start':self.register()
        def next(self):
            events.append(('next',len(self),self.data.close[0]))
            if phase=='runtime' and len(self)==1:self.register()
        def notify_timer(self,timer,when,**kwargs):received('strategy',timer,when,**kwargs)
    engine=App(stdstats=False,runonce=False)
    # Existing timer ensures a runtime registration is not hidden by an empty
    # scheduler: fixed BT still leaves the new handle unstarted in this run.
    engine.add_timer(time(9,3),strats=False,token='baseline')
    f=pd.DataFrame({n:[10.,20.,30.,40.,50.,60.] for n in ['open','high','low','close','volume']},index=pd.date_range('2026-01-02 09:00',periods=6,freq='min'))
    engine.adddata(api.feeds.PandasData(dataname=f,timeframe=api.TimeFrame.Minutes));engine.addstrategy(Business);engine.run()
    return handles,events


@pytest.mark.parametrize('owner,strats',[('strategy',False),('cerebro',False),('cerebro',True)])
@pytest.mark.parametrize('phase',['init','start','runtime'])
@pytest.mark.parametrize('cheat',[False,True])
def test_strategy_and_cerebro_timer_owner_init_start_and_runtime_registration_match_bt(owner,strats,phase,cheat):
    assert profile(tl,owner,phase,cheat,strats)==profile(official,owner,phase,cheat,strats)
