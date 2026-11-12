"""Generic public binding: no account connection or application profile."""

from dataclasses import replace

import pytest
from nautilus_trader.config import DataClientConfig, ExecutionClientConfig
from nautilus_trader.live import LiveNodeConfig
from nautilus_trader.model import TraderId
from tl_store.ctrader import CTraderConfig
from tl_store.ctrader.store import CTraderStoreConfig, build_adapter

from tradelearn_runtime.store import NautilusStore


def assembled(config):
    return NautilusStore(adapter="ctrader", config=config)


def config(path, login=100001):
    return CTraderStoreConfig(
        options=CTraderConfig(
            expected_login=login, symbols=("BTCUSD",), asset_classes=("CRYPTOCURRENCY",)
        ),
        node_config=LiveNodeConfig(
            trader_id=TraderId("CLIENT-001"),
            data_clients={"CTRADER": DataClientConfig()},
            exec_clients={"CTRADER": ExecutionClientConfig()},
        ),
        state_dir=path,
    )


def test_constructs_isolated_without_acquire_or_state_writes(tmp_path, monkeypatch):
    from tl_store.ctrader.session import CTraderAccountSession

    async def forbidden(*args, **kwargs):
        raise AssertionError("connected during assembly")

    monkeypatch.setattr(CTraderAccountSession, "acquire", forbidden)
    a, b = assembled(config(tmp_path)), assembled(config(tmp_path, 100002))
    assert a.account_id == "CTRADER-100001" and b.account_id == "CTRADER-100002"
    assert a.state_path != b.state_path and a.getbroker() is not b.getbroker()
    da = a.data_factories["CTRADER"]
    ea = a.exec_factories["CTRADER"]
    assert da.session is ea.session
    assert da.session is not b.data_factories["CTRADER"].session
    assert not list(tmp_path.iterdir())
    feed = a.getdata("BTCUSD.CTRADER-1-SECOND-MID-INTERNAL")
    assert feed.bar_type.is_internally_aggregated()
    with pytest.raises(RuntimeError, match="not started"):
        a.getbroker().getcash()


@pytest.mark.parametrize(
    "bar",
    [
        "BTCUSD.CTRADER-1-SECOND-MID-EXTERNAL",
        "BTCUSD.CTRADER-1-SECOND-LAST-INTERNAL",
        "OTHER.CTRADER-1-SECOND-MID-INTERNAL",
        "BTCUSD.OTHER-1-SECOND-MID-INTERNAL",
    ],
)
def test_refuses_unsupported_feed_before_start(tmp_path, bar):
    a = assembled(config(tmp_path))
    with pytest.raises(ValueError):
        a.getdata(bar)


def test_rejects_wrong_config_and_unknown_parameters(tmp_path):
    with pytest.raises(TypeError):
        build_adapter({})
    with pytest.raises(TypeError):
        CTraderStoreConfig(options=None, node_config=None, unknown=True)
    with pytest.raises(ValueError, match="CTRADER"):
        replace(config(tmp_path), node_config=LiveNodeConfig(trader_id=TraderId("CLIENT-001")))


def test_real_official_node_assembles_installed_factories_offline(tmp_path, monkeypatch):
    from tl_store.ctrader.session import CTraderAccountSession

    async def forbidden(*args, **kwargs):
        raise AssertionError("broker access")

    monkeypatch.setattr(CTraderAccountSession, "acquire", forbidden)
    store = assembled(config(tmp_path))
    node = store._build_node()
    try:
        assert store._host is None
    finally:
        node.dispose()


def test_installed_entrypoint_runs_exactly_one_shared_driver(tmp_path, monkeypatch):
    from importlib.metadata import entry_points

    from tradelearn import engine as bt
    from tradelearn_runtime._adapters import _factories

    monkeypatch.delitem(_factories, "ctrader", raising=False)
    assert len(list(entry_points(group="tradelearn.store_adapters", name="ctrader"))) == 1
    store = __import__("tradelearn_runtime", fromlist=["NautilusStore"]).NautilusStore(
        adapter="ctrader", config=config(tmp_path)
    )
    engine = bt.Cerebro()
    engine.setbroker(store.getbroker())
    engine.adddata(store.getdata("BTCUSD.CTRADER-1-SECOND-MID-INTERNAL"))
    engine.addstrategy(bt.Strategy)
    calls = []

    async def run(current):
        calls.append(current)
        assert current._active_store_driver is store
        raise RuntimeError("offline startup failure")

    monkeypatch.setattr(store, "_run_native", run)
    with pytest.raises(RuntimeError, match="offline startup failure"):
        engine.run()
    assert calls == [engine] and engine._active_store_driver is None
    assert [event[0] for event in store.get_notifications()] == ["STARTING", "FAILED"]
    with pytest.raises(RuntimeError, match="single use"):
        engine.run()
