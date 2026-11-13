"""Engine keeps BT input float precision; native venue precision is separate."""
import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as bt


@pytest.mark.parametrize('on_close',[False,True])
@pytest.mark.parametrize('size',[.125,1e-8])
def test_market_size_price_fee_and_account_keep_fixed_bt_float_semantics(on_close,size):
    prices=[100.123456789123,105.21052631578948,110.47368421052632,111.987654321987]
    frame=pd.DataFrame({name:prices for name in ('open','high','low','close')},index=pd.date_range('2026-01-01',periods=4));frame['volume']=100.
    results=[]
    for api in (official,bt):
        notices=[]
        class Business(api.Strategy):
            def next(self):
                if len(self)==1:self.buy(size=size)
            def notify_order(self,order):
                if order.status==order.Completed:
                    notices.append((order.executed.size,order.executed.price,order.executed.comm,order.executed.dt,self.broker.getcash(),self.broker.getvalue()))
        engine=api.Cerebro(stdstats=False);engine.broker.setcash(10000.)
        engine.broker.setcommission(commission=.00123456789)
        engine.broker.set_coc(on_close)
        engine.adddata(api.feeds.PandasData(dataname=frame));engine.addstrategy(Business)
        strategy=engine.run()[0]
        assert len(notices)==1
        results.append((notices[0],strategy.broker.getcash(),strategy.broker.getvalue()))
    assert results[1][0]==pytest.approx(results[0][0],rel=1e-12,abs=1e-12)
    assert results[1][1:]==pytest.approx(results[0][1:],rel=1e-12,abs=1e-12)
