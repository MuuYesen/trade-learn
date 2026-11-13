"""Optional live fund facts are projections of the existing trusted account source."""
from types import SimpleNamespace as NS
import pytest
from tradelearn_runtime.broker import NautilusBroker


def broker(snapshot):
    return NautilusBroker(NS(account=lambda: snapshot))


def test_standard_fund_getters_read_source_values_not_equity_ratios():
    b=broker(dict(cash=70.,equity=120.,fundmode=True,fundvalue=37.5,fundshares=3.2))
    assert b.fundmode is True and b.get_fundmode() is True
    assert b.fundvalue==b.get_fundvalue()==37.5
    assert b.fundshares==b.get_fundshares()==3.2
    assert b._fund_notification_facts()==(70.,120.,37.5,3.2)


def test_missing_live_fund_source_does_not_fabricate_units_or_break_normal_account():
    b=broker(dict(cash=70.,equity=120.))
    assert b.getcash()==70. and b.getvalue()==120.
    assert b.fundmode is False
    assert b._fund_notification_facts() is None
    with pytest.raises(ValueError,match='authoritative.*fund'):
        b.get_fundvalue()
    with pytest.raises(ValueError,match='authoritative.*fund'):
        b.get_fundshares()


@pytest.mark.parametrize('change',[
    dict(fundvalue=float('nan')),dict(fundshares=float('inf')),
    dict(fundvalue=True),dict(fundshares=False),dict(fundmode=1),
    dict(cash=float('inf')),dict(equity=None),dict(fundvalue=None),
])
def test_bad_or_partial_source_is_not_silently_replaced(change):
    facts=dict(cash=70.,equity=120.,fundmode=True,fundvalue=37.5,fundshares=3.2)
    facts.update(change)
    with pytest.raises(ValueError,match='fund|account'):
        broker(facts)._fund_notification_facts()


def test_zero_and_negative_nav_are_preserved_not_clamped():
    for nav in (0.,-3.5):
        b=broker(dict(cash=0.,equity=nav*4,fundmode=True,fundvalue=nav,fundshares=4.))
        assert b.fundvalue==nav
        assert b._fund_notification_facts()==(0.,nav*4,nav,4.)


def test_fund_notification_reads_one_atomic_provider_snapshot():
    calls=[]
    def account():
        calls.append(None)
        return dict(cash=70.,equity=120.,fundmode=True,fundvalue=37.5,fundshares=3.2)
    b=NautilusBroker(NS(account=account))
    assert b._fund_notification_facts()==(70.,120.,37.5,3.2)
    assert len(calls)==1


@pytest.mark.parametrize('has_fund',[False,True])
def test_actual_na_callback_reads_optional_fund_source_on_the_one_runtime(tmp_path,monkeypatch,has_fund):
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tradelearn import engine as bt
    from tests.unit.engine.test_nautilus_store import store
    instrument=TestInstrumentProvider.btcusdt_perp_binance()
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],
        fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    bar_type=BarType.from_str(str(instrument.id)+'-1-SECOND-LAST-EXTERNAL')
    native.add_data([Bar(bar_type,*[instrument.make_price(1000)]*4,instrument.make_qty(10),
        (i+1)*1000000000,(i+1)*1000000000) for i in range(3)])
    provider_calls=[];notices=[];errors=[];rows=[]
    def provider(cache,portfolio):
        # Explicit synthetic fund-source values, not venue balance-derived NAV.
        assert cache.account_for_venue(instrument.id.venue) is not None
        provider_calls.append(None)
        snapshot=dict(cash=70.,equity=120.)
        if has_fund:snapshot.update(fundmode=True,fundvalue=37.5,fundshares=3.2)
        return snapshot
    class Business(bt.Strategy):
        def next(self):rows.append((len(self),self.broker.getvalue()))
    if has_fund:
        def notify_fund(self,*facts):notices.append(facts)
        Business.notify_fund=notify_fund
    value=store(tmp_path,account_provider=provider)
    engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker())
    engine.adddata(value.getdata(bar_type));engine.addstrategy(Business)
    if has_fund:
        engine.addanalyzer(bt.analyzers.DrawDown,_name='dd',fund=True)
        engine.addanalyzer(bt.analyzers.Returns,_name='returns',fund=True)
    class ReplayNode:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return NS(stop=lambda:None)
        async def run_async(self):native.run()
        def dispose(self):
            errors.append(self.actor._worker_error);native.dispose();self.actor=None
    monkeypatch.setattr(value,'_build_node',ReplayNode)
    [business]=engine.run()
    assert errors==[None] and rows==[(1,120.),(2,120.),(3,120.)]
    assert notices==([(70.,120.,37.5,3.2)]*3 if has_fund else [])
    assert business._runtime is value._host.frontend
    if has_fund:
        assert business.analyzers.dd.get_analysis()['max']['drawdown']==0.
        assert business.analyzers.returns.get_analysis()['rtot']==0.
