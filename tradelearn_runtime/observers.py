"""External observation assembly; SDK identity and transport stay with the SDK."""


def attach_runtime_observer(runtime, factory):
    if factory is None:
        return None
    observer = factory(runtime)
    if callable(observer):
        return runtime.add_observer(observer)
    from tradelearn.core.strategy_events import ObserverSpec
    if not isinstance(observer, ObserverSpec):
        raise TypeError('runtime observer factory must return callable or ObserverSpec')
    return runtime.add_observer(observer.observer, events=observer.events,
        phases=observer.phases, context=observer.context,
        error_handler=observer.error_handler)
