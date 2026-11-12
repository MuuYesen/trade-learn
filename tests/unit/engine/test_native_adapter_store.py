from tests.unit.backtest._engine_config import feed as _as_feed
"""Installed adapter binding reuses the native driver, without polling."""
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn_runtime import NautilusStore
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.config import DataClientConfig, ExecutionClientConfig
from nautilus_trader.live.clients import DataClientFactory, ExecutionClientFactory
from nautilus_trader.model import TraderId
from tradelearn import engine as bt
from tradelearn_runtime.store import NautilusStore

class DataFactory(DataClientFactory):
    def create(self, **kwargs):
        raise AssertionError('construction must not create clients')
class ExecFactory(ExecutionClientFactory):
    def create(self, **kwargs):
        raise AssertionError('construction must not create clients')

def native_binding(options):
    path, account = options
    config=LiveNodeConfig(trader_id=TraderId('CLIENT-001'),
        data_clients={'TEST':DataClientConfig()},exec_clients={'TEST':ExecutionClientConfig()})
    return dict(api_version=2,config=config,data_factories={'TEST':DataFactory()},
        exec_factories={'TEST':ExecFactory()},account_id=account,state_dir=path)

def native(path, account='TEST-001'):
    NautilusStore.register_adapter('native-test', native_binding)
    return NautilusStore(adapter='native-test', config=(path, account))

def test_installed_native_binding_is_inert_isolated_and_uses_one_driver(tmp_path, monkeypatch):
    from tradelearn_runtime._adapters import _factories
    a,b=native(tmp_path),native(tmp_path,'TEST-002')
    assert a.state_path!=b.state_path and a.getbroker() is not b.getbroker()
    assert not list(tmp_path.iterdir())
    store=a
    engine=bt.Cerebro();engine.setbroker(store.getbroker())
    feed=store.getdata('BTCUSD.TEST-1-SECOND-MID-INTERNAL');engine.adddata(_as_feed(feed))
    engine.addstrategy(bt.Strategy)
    assert feed._store is store and store.getbroker() is a.getbroker()
    calls=[]
    async def run(current):
        calls.append(current)
        assert current._active_store_driver is a
        return ['native-result']
    monkeypatch.setattr(a,'_run_native',run)
    assert engine.run()==['native-result'] and calls==[engine]
    assert engine._active_store_driver is None
    assert [r[0] for r in store.get_notifications()]==['STARTING','STOPPED']
    with pytest.raises(RuntimeError,match='single use'):engine.run()

def test_external_factory_not_allowed_through_official_store(tmp_path):
    a=native(tmp_path)
    from nautilus_trader.adapters.binance import BinanceExecutionClientConfig
    from nautilus_trader.model import AccountId
    config=LiveNodeConfig(trader_id=TraderId('CLIENT-001'),data_clients={'TEST':DataClientConfig()},
        exec_clients={'TEST':BinanceExecutionClientConfig(account_id=AccountId('BINANCE-001'))})
    with pytest.raises(ValueError,match='official'):
        NautilusStore(config,data_factories=a.data_factories,exec_factories=a.exec_factories)

def test_native_binding_requires_explicit_account_and_real_factory_contract(tmp_path):
    a=native(tmp_path)
    with pytest.raises(ValueError,match='account'):
        NautilusStore(a.config,data_factories=a.data_factories,exec_factories=a.exec_factories)
    def malformed(config):
        result=native_binding((tmp_path, 'TEST-001'));result['data_factories']={'TEST':object()};return result
    NautilusStore.register_adapter('bad-native', malformed)
    with pytest.raises(TypeError,match='factory'):
        NautilusStore(adapter='bad-native', config=None)

@pytest.mark.parametrize('field,value,exception', [
    ('api_version', 1, TypeError),
    ('unexpected', True, ValueError),
    ('account_id', 'BAD', ValueError),
])
def test_independent_assembly_refuses_bad_contract_before_runtime(tmp_path, field, value, exception):
    def invalid(config):
        result = native_binding((tmp_path, 'TEST-001'))
        result[field] = value
        return result
    from tradelearn_runtime._adapters import _factories
    old = _factories.get('invalid-contract')
    _factories['invalid-contract'] = invalid
    try:
        with pytest.raises(exception):
            NautilusStore(adapter='invalid-contract', config=None)
        assert not list(tmp_path.iterdir())
    finally:
        if old is None: _factories.pop('invalid-contract', None)
        else: _factories['invalid-contract'] = old


def test_explicit_native_config_and_adapter_selection_cannot_overlap(tmp_path):
    value = native(tmp_path)
    with pytest.raises(ValueError, match='owns native assembly'):
        NautilusStore(adapter='native-test', config=(tmp_path, 'TEST-001'),
                      data_factories=value.data_factories)
