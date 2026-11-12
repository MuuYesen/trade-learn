"""Explicitly qualified SDK/TL observation versions; no trading or cloud client."""
from types import SimpleNamespace

import pytest

from tests.windows_private_directory import make_private_test_directory
from leafquant_monitor import MonitorSettings, TradeLogger
from leafquant_monitor import tradelearn as sdk


@pytest.mark.parametrize('version', ['0.2.9', '0.3.0', '0.3.1'])
def test_qualified_versions_assemble_real_local_logger(version, monkeypatch, tmp_path):
    make_private_test_directory(tmp_path)
    monkeypatch.setattr(sdk.tradelearn, '__version__', version)
    monkeypatch.setattr(sdk, 'OBSERVER_API_VERSION', 1)
    settings = MonitorSettings('LOCAL', 'compat', 'fixture-account', str(tmp_path), account_source='external')
    logger = TradeLogger(settings)
    config = sdk.TradeLogConfig(context=dict(environment='paper', run_id='fixture-context',
        source='LOCAL', strategy_id='compat', account_id='fixture-account', state_dir=str(tmp_path)), logger=logger)
    strategy = SimpleNamespace(monitor_config=config)
    try:
        spec = sdk._assemble(strategy)
        assert spec.context['source'] == 'LOCAL'
        assert strategy.monitor_logger is logger and logger.sender is None
    finally:
        logger.close()


@pytest.mark.parametrize('version,api', [('0.2.8',1),('0.3.2',1),('0.4.0',1),('0.3.0+unknown',1),('0.2.9',2),('0.3.0',2),('0.3.1',2),('0.3.1+unknown',1)])
def test_unknown_version_or_api_rejected_before_logger_factory(version, api, monkeypatch):
    monkeypatch.setattr(sdk.tradelearn, '__version__', version)
    monkeypatch.setattr(sdk, 'OBSERVER_API_VERSION', api)
    def forbidden(*args, **kwargs):
        pytest.fail('incompatible observation contract acquired a recorder')
    monkeypatch.setattr(sdk, 'TradeLogger', forbidden)
    with pytest.raises(RuntimeError, match='observer API v1'):
        sdk._assemble(SimpleNamespace())
