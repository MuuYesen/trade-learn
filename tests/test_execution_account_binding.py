"""Account identity is independent of optional logging and cannot cross routing."""
from types import SimpleNamespace as NS, MethodType
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.model import AccountId, InstrumentId, StrategyId
from tradelearn_runtime.execution import ExecutionHost


def host(account='SIM-001', routed='SIM-001'):
    iid=InstrumentId.from_str('EURUSD.SIM')
    value=NS(_account_id=account,_instrument_ids={'EURUSD':iid},strategy_id=StrategyId('TL-001'),
        cache=NS(account=lambda _:NS(id=AccountId('SIM-001')),
                 account_for_venue=lambda _:NS(id=AccountId(routed))))
    value._assert_account_binding=MethodType(ExecutionHost._assert_account_binding,value)
    value._owns_native_order=MethodType(ExecutionHost._owns_native_order,value)
    return value


@pytest.mark.parametrize('account,routed',[('WRONG-001','SIM-001'),('SIM-001','OTHER-001'),(None,'SIM-001')])
def test_wrong_or_missing_account_cannot_pass_execution_admission(account,routed):
    with pytest.raises(ValueError,match='account'):
        host(account,routed)._assert_account_binding()


def test_other_account_order_and_position_never_reach_business_projection():
    value=host()
    value.cache.order=lambda _:NS(account_id=AccountId('OTHER-001'),strategy_id=value.strategy_id,
        instrument_id=value._instrument_ids['EURUSD'])
    value.frontend_broker=NS(apply=lambda *a,**k:pytest.fail('foreign order projected'))
    ExecutionHost._project_order(value,NS(client_order_id='foreign'))
    value._position_events=set()
    ExecutionHost._project_position(value,NS(account_id=AccountId('OTHER-001'),strategy_id=value.strategy_id))
    assert not value._position_events


def test_same_account_foreign_strategy_position_never_reaches_business():
    value=host();value._position_events=set()
    ExecutionHost._project_position(value,NS(account_id=AccountId('SIM-001'),strategy_id=StrategyId('OTHER-001')))
    assert not value._position_events


@pytest.mark.parametrize('controls',[{'client_id':'OTHER'},{'params':{'account':'OTHER'}},{'position_id':'FOREIGN'}])
def test_per_order_override_or_foreign_position_never_reaches_order_factory(controls):
    value=host()
    value._bound_position=MethodType(ExecutionHost._bound_position,value)
    value.cache.position=lambda _:NS(account_id=AccountId('OTHER-001'),strategy_id=value.strategy_id,
        instrument_id=value._instrument_ids['EURUSD'])
    value.cache.instrument=lambda _:pytest.fail('invalid controls reached order construction')
    with pytest.raises(ValueError):ExecutionHost._make_order(value,NS(symbol='EURUSD'),controls)


@pytest.mark.parametrize('method',['submit_order','submit_order_list','modify_order','modify_orders',
    'cancel_order','cancel_orders','cancel_all_orders','close_position','close_all_positions'])
def test_all_native_execution_entrypoints_reject_client_routing_override(method):
    value=host();value._native_route=MethodType(ExecutionHost._native_route,value)
    value._before_native_submission=MethodType(ExecutionHost._before_native_submission,value)
    with pytest.raises(ValueError,match='overrides'):
        getattr(ExecutionHost,method)(value,[None] if method=='submit_order_list' else None,client_id='OTHER')
