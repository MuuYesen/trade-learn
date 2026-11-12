"""Internal legacy execution configurations used by semantic regression tests."""

def configured_engine(**options):
    """Internal TL execution configuration outside the strict BT facade.

    Convenience frontends and tests normalize legacy TL-only settings here.
    Users should configure the broker directly for cash/commission/COC.
    """
    from tradelearn.engine.cerebro import Cerebro
    engine = Cerebro()
    engine._initialize(**options)
    engine.p = engine.params = __import__("tradelearn.engine.base", fromlist=["Params"]).Params(Cerebro._bt_defaults)
    return engine


def feed(data):
    """Build an actual BT-style feed for semantic test fixtures."""
    import pandas as pd
    from tradelearn.engine.feeds import PandasData
    if not isinstance(data,pd.DataFrame):
        return data
    common={'open','high','low','close','volume','openinterest','datetime'}
    extra=tuple(str(c).lower() for c in data.columns if str(c).lower() not in common)
    if extra:
        cls=type('FixtureData',(PandasData,),{'lines':extra,'params':tuple((name,-1) for name in extra)})
        return cls(dataname=data)
    return PandasData(dataname=data)
