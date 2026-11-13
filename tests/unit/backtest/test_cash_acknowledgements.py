"""Fund-unit metadata consumes only exact outstanding native cash facts."""
import pytest
from tradelearn.backtest.broker import RustBroker


@pytest.mark.parametrize('event',[(True,'add',100.,10100.),(9,'add',100.,10100.),(1,'set',100.,10100.),(1,'add',101.,10101.),(1,'add',100.,float('nan')),(1,'add',100.,True)])
def test_wrong_cash_ack_does_not_mutate_units_or_forget_the_outstanding_command(event):
    b=RustBroker(cash=10000);b.add_cash(100.)
    with pytest.raises(ValueError):b.receive_cash_events([event])
    assert b.fundshares==100. and b._pending_cash_requests=={1:('add',100.)}


def test_cash_ack_batch_is_atomic_and_replay_never_adds_fund_units_twice():
    b=RustBroker(cash=10000);b.add_cash(100.);b.add_cash(0.)
    with pytest.raises(ValueError):b.receive_cash_events([(1,'add',100.,10100.),(1,'add',100.,10100.)])
    assert b.fundshares==100. and len(b._pending_cash_requests)==2
    b.receive_cash_events([(1,'add',100.,10100.),(2,'add',0.,10100.)])
    assert b.fundshares==101. and not b._pending_cash_requests
    with pytest.raises(ValueError):b.receive_cash_events([(1,'add',100.,10100.)])
    assert b.fundshares==101.
