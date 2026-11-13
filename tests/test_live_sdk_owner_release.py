"""A retained closed logger must not keep an unsendable native Strategy alive."""

import gc
import weakref

import pytest

pytest.importorskip('nautilus_trader')
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import StrategyId
from leafquant_monitor import MonitorSettings
from leafquant_monitor.strategy import TradeLogStrategy


def test_native_sdk_dispose_releases_bound_clock_owner_without_changing_clock_domain(tmp_path):
    enabled=gc.isenabled()
    logger=None
    try:
        gc.collect();gc.disable()
        actor=TradeLogStrategy(StrategyConfig(strategy_id=StrategyId('TL-001')),
            monitor=MonitorSettings('LOCAL','owner-fixture','fixture-account',str(tmp_path),
                                    account_source='external',framework_strategy_id='TL-001'))
        logger=actor.logger
        owner=weakref.ref(actor)
        actor.on_dispose()
        del actor
        retained=owner() is not None
        assert not retained, 'closed SDK logger retains the native owner through its clock callback'
        from tradelearn.core.observation_clock import monotonic_ns
        assert logger._monotonic_ns is monotonic_ns
    finally:
        logger=None
        gc.collect()
        if enabled:gc.enable()
