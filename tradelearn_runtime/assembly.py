"""Select a single explicit native host without importing optional NA."""

def select_driver(engine):
    """Select one explicit store driver before the polling runtime is created."""
    candidates = [*engine.stores, getattr(engine.broker, '_store', None),
                  *(getattr(data, '_store', None) for data in engine.datas)]
    drivers = []
    for value in candidates:
        if value is not None and callable(getattr(value, '_run_cerebro', None)):
            if all(value is not other for other in drivers):
                drivers.append(value)
    if len(drivers) > 1:
        raise ValueError('Cerebro requires one store driver; accounts need separate engines')
    return drivers[0] if drivers else None
