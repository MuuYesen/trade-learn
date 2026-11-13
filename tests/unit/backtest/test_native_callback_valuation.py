"""The existing native runners release the owner loan across Python business."""
import numpy as np
import pytest
from tradelearn._rust import RustBacktestEngine,RustBarRunner,RustClockedMultiDataRunner


@pytest.mark.parametrize('runner_kind',['engine','bar_runner','clocked_multi'])
def test_native_callback_can_observe_and_settle_same_native_account_without_reentrant_mutable_loan(runner_kind):
    times=np.array([1,2,3],dtype=np.int64);values=np.array([10.,10.,10.]);engine=RustBacktestEngine(times,values,values,values,values,values,1000.,0.,False,False,False,0.,0.,True,True,False)
    class Broker:
        def drain_cash_controls(self):return []
        def receive_cash_events(self,events):raise AssertionError('callback drains its actual explicit valuation ACK')
    rows=[]
    def on_bar(*args):
        cursor=args[0]
        if cursor==0:
            engine.apply_cash_controls([(1,'add',500.)])
            assert engine.observe_valuation(None,True,True)==(1500.,1500.,0.)
            assert engine.drain_cash_events()==[(1,'add',500.,1500.)]
        rows.append(engine.observe_valuation(None,True,False))
    if runner_kind=='engine':engine.run_bar_loop(Broker(),on_bar,0,3)
    elif runner_kind=='bar_runner':RustBarRunner(times,[]).run(engine,Broker(),on_bar,0,3)
    else:RustClockedMultiDataRunner(['data0'],[times],[values],[values],[values],[values],[values]).run(engine,Broker(),on_bar,0,3)
    assert rows==[(1500.,1500.,0.)]*3
