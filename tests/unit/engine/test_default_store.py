"""DefaultStore exercises the driver contract without any optional adapter."""
from types import SimpleNamespace
import pytest
from tradelearn import engine as bt

class Binding:
    def __init__(self, config):
        self.config = config
        self.broker = SimpleNamespace()
        self.feeds = []
    def getdata(self, store, *, name):
        feed = SimpleNamespace(_store=store, _name=name, is_live=True)
        self.feeds.append(feed)
        return feed
    def validate(self, store, engine):
        pass
    def run(self, store):
        assert store.getbroker() is self.broker
        store._host = SimpleNamespace(frontend=SimpleNamespace(strategy=self.config, failed=False))
    def runtime(self, host):
        return host.frontend if host else None


def test_default_store_registered_instances_are_isolated_and_inert(monkeypatch):
    from tradelearn.engine.stores.default import _factories
    monkeypatch.setitem(_factories, 'offline-test', Binding)
    stores = [bt.stores.DefaultStore(adapter='offline-test', config=object()) for _ in range(2)]
    assert stores[0].getbroker() is not stores[1].getbroker()
    assert stores[0]._host is None
    engine = bt.Cerebro()
    store = stores[0]
    engine.setbroker(store.getbroker())
    engine.adddata(store.getdata(name='bar'))
    engine.addstrategy(bt.Strategy)
    assert engine.run() == [store._adapter.config]
    assert [n[0] for n in store.get_notifications()] == ['STARTING', 'STOPPED']
    with pytest.raises(RuntimeError, match='single use'):
        engine.run()


def test_default_store_rejects_missing_adapter_and_unknown_parameters():
    with pytest.raises(ValueError, match='not installed'):
        bt.stores.DefaultStore(adapter='missing-test', config={})
    with pytest.raises(TypeError):
        bt.stores.DefaultStore(adapter='missing-test', config={}, account='duplicate')


def test_registration_refuses_accidental_override(monkeypatch):
    from tradelearn.engine.stores.default import _factories
    monkeypatch.setitem(_factories, 'offline-test', Binding)
    with pytest.raises(ValueError, match='already registered'):
        bt.stores.register_adapter('offline-test', lambda config: None)


def test_default_store_preserves_failure_and_closes_delivery(monkeypatch):
    from tradelearn.engine.stores.default import _factories
    events = []
    class FailedBinding(Binding):
        def run(self, store):
            def finish():
                events.append('delivery closed')
                raise OSError('cleanup failure')
            store._host = SimpleNamespace(logger=SimpleNamespace(finish_delivery=finish))
            raise ValueError('primary failure')
    monkeypatch.setitem(_factories, 'offline-test', FailedBinding)
    store = bt.stores.DefaultStore(adapter='offline-test', config=None)
    engine = bt.Cerebro()
    engine.setbroker(store.getbroker())
    engine.adddata(store.getdata(name='bar'))
    engine.addstrategy(bt.Strategy)
    with pytest.raises(ValueError, match='primary failure'):
        engine.run()
    assert events == ['delivery closed']
    assert engine._active_store_driver is None
    assert store.get_notifications()[-1][0] == 'FAILED'


def test_installed_adapter_entrypoint_is_loaded_only_when_selected(monkeypatch):
    import tradelearn.engine.stores.default as module
    calls = []
    class Entry:
        def load(self):
            calls.append('loaded')
            return Binding
    monkeypatch.setattr(module, 'entry_points', lambda **kwargs: [Entry()])
    monkeypatch.setattr(module, '_factories', {})
    assert calls == []
    store = bt.stores.DefaultStore(adapter='installed-test', config='account')
    assert calls == ['loaded']
    assert store._host is None


def test_default_store_cannot_mix_instances(monkeypatch):
    from tradelearn.engine.stores.default import _factories
    monkeypatch.setitem(_factories, 'offline-test', Binding)
    a, b = [bt.stores.DefaultStore(adapter='offline-test', config=None) for _ in range(2)]
    engine = bt.Cerebro()
    engine.setbroker(a.getbroker())
    engine.adddata(b.getdata(name='bar'))
    engine.addstrategy(bt.Strategy)
    with pytest.raises(ValueError, match='one store driver'):
        engine.run()
