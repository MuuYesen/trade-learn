"""The public Store namespace is inert; Cerebro owns the sole launch entry."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_engine_store_namespace_does_not_load_optional_native_packages():
    source = Path(__file__).resolve().parents[3]
    script = '''
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('nautilus_trader', 'tl_store', 'boto3', 'botocore', 'tradelearn_runtime'):
            raise AssertionError('optional dependency imported: ' + fullname)
sys.meta_path.insert(0, NoNative())
import tradelearn.engine as bt
assert 'NautilusStore' in bt.stores.__all__
assert 'NautilusStore' in dir(bt.stores)
assert 'tradelearn_runtime' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=source,
                            env={**os.environ, 'PYTHONPATH': str(source)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_both_public_store_imports_use_the_existing_implementation():
    pytest.importorskip('nautilus_trader')
    import tradelearn.engine as bt
    from tradelearn.engine.stores import NautilusStore
    from tradelearn_runtime import NautilusStore as InternalStore
    assert bt.stores.NautilusStore is NautilusStore is InternalStore
    assert not hasattr(NautilusStore, 'run')


def test_cerebro_monitor_without_native_store_is_rejected_before_state():
    from tradelearn import engine as bt
    with pytest.raises(ValueError, match='Store-owned'):
        bt.Cerebro().run(monitor=True)


def test_native_cerebro_applies_public_run_options_before_driver(tmp_path, monkeypatch):
    pytest.importorskip('nautilus_trader')
    from tests.unit.engine.test_nautilus_store import store, engine_for, FakeNode
    value = store(tmp_path)
    engine = engine_for(value)
    monkeypatch.setattr(value, '_build_node', lambda: FakeNode([]))
    assert engine.run(stdstats=False, runonce=False) == []
    assert engine.stdstats is False and engine.p.runonce is False


def test_unverified_monitor_retains_local_log_without_sender(tmp_path, monkeypatch):
    pytest.importorskip('nautilus_trader')
    from tests.unit.engine.test_nautilus_store import store, engine_for, FakeNode
    import leafquant_monitor.bootstrap as bootstrap
    value = store(tmp_path)
    engine = engine_for(value)
    monkeypatch.setattr(value, '_build_node', lambda: FakeNode([]))
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', lambda: pytest.fail('unverified binding loaded AWS'))
    with pytest.warns(RuntimeWarning, match='local logs retained'):
        engine.run(monitor={
            'endpoint': 'https://sqs.ap-southeast-1.amazonaws.com/123456789012/customer.fifo',
            'credentials': {'access_key_id': 'synthetic-key', 'secret_access_key': 'synthetic-secret'},
        })
    assert value._host.logger.sender is None
    assert value._monitor_error == 'TRUSTED_MONITOR_BINDING_UNAVAILABLE'
    logs = list((value.state_path/'logs'/'segments').glob('*.log'))
    assert logs and any(b'TRUSTED_MONITOR_BINDING_UNAVAILABLE' in p.read_bytes() for p in logs)
    assert all(b'synthetic-secret' not in p.read_bytes() for p in logs)
    assert value._remote_monitor._closed and not value._remote_monitor._credentials


def test_final_user_template_constructs_and_uses_one_native_entry(tmp_path, monkeypatch):
    pytest.importorskip('nautilus_trader')
    pytest.importorskip('tl_store.ctrader')
    import runpy
    from tradelearn.engine.stores import NautilusStore
    from tests.unit.engine.test_nautilus_store import FakeNode
    path = Path(__file__).resolve().parents[3]/'examples/live/ctrader_monitor.py'
    calls = []
    def node(value):
        calls.append(value)
        return FakeNode([])
    monkeypatch.setattr(NautilusStore, '_build_node', node)
    monkeypatch.chdir(tmp_path)
    with pytest.warns(RuntimeWarning, match='local logs retained'):
        namespace=runpy.run_path(str(path),run_name='__main__')
    assert len(calls) == 1
    value = calls[0]
    assert value._used and value._host.logger.sender is None
    assert value._engine.broker is value.getbroker()
    assert len(value._engine.datas) == 1
    assert value._engine.strats[0][0] is namespace['MyStrategy']
    assert value._remote_monitor._closed

    assert value._engine.stdstats is False
    assert str(value.config.trader_id)=='DEMO-001'
    assert str(value.strategy_config.strategy_id)=='DEMO-001'
    assert str(value._engine.datas[0].bar_type)=='BTCUSD.CTRADER-1-SECOND-MID-INTERNAL'
    keys=value.data_factories['CTRADER'].options.credentials
    assert keys.access_token=='YOUR_CTRADER_ACCESS_TOKEN'
    assert keys.refresh_token=='YOUR_CTRADER_REFRESH_TOKEN'
