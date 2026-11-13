"""Replay equity is a native account observation, independent of stdstats."""
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


def profile(api,clone,coc,stdstats,flags):
    values=[10.,30.,90.,70.,120.,50.,150.,80.]
    f=pd.DataFrame(dict(open=values,high=[v+5 for v in values],low=[v-3 for v in values],
                       close=[v+2 for v in values],volume=range(1,9)),index=pd.to_datetime([
        '2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:08','2026-01-02 09:09',
        '2026-01-02 09:16','2026-01-02 09:22','2026-01-02 09:23','2026-01-02 09:31']))
    rows=[];observed=[];orders=[]
    class Business(api.Strategy):
        def __init__(self):self.count=0;self.sent=False;self.closed=False
        def next(self):
            self.count+=1;d=self.datas[-1]
            rows.append((self.broker.getcash(),self.broker.getvalue()))
            if self.count==3:self.buy(data=d,size=.125);self.sent=True
            elif self.sent and not self.closed and self.getposition(d).size and len(d)>=3:
                self.close(data=d);self.closed=True
        def notify_order(self,o):
            orders.append((o.getstatusname(),o.created.dt,o.executed.dt,o.executed.size,o.executed.price,o.executed.comm))
    class Audit(api.Observer):
        lines=('equity',)
        _stclock=True
        def next(self):
            observed.append(self._owner.broker.getvalue());self.lines.equity[0]=observed[-1]
    c=api.Cerebro(stdstats=stdstats,runonce=False)
    c.broker.setcash(12500);c.broker.setcommission(commission=.002);c.broker.set_coc(coc)
    raw=api.feeds.PandasData(dataname=f,timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(raw)
    c.replaydata(raw,timeframe=api.TimeFrame.Minutes,compression=7,**flags)
    c.addstrategy(Business);c.addobserver(Audit);c.run()
    return dict(rows=rows,observer=observed,orders=orders),list(c.stats.equity) if api is tl else None


@pytest.mark.parametrize('stdstats',[False,True])
@pytest.mark.parametrize('coc',[False,True])
@pytest.mark.parametrize('clone,flags',[(False,dict(rightedge=False,boundoff=2)),(True,dict(bar2edge=False)),(True,dict(adjbartime=True))])
def test_native_replay_account_equity_business_observer_and_report_share_bt_phase(stdstats,coc,clone,flags):
    actual,report=profile(tl,clone,coc,stdstats,flags);expected,_=profile(official,clone,coc,stdstats,flags)
    assert actual==expected
    assert report==expected['observer']
