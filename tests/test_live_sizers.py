"""BT Sizer protocol on the same native broker facts; never a second broker."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn import engine as bt
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.components import native_sizer, validate_components


def test_custom_sizer_receives_bound_strategy_authoritative_cash_and_explicit_policy():
    seen=[]
    class Custom(bt.Sizer):
        params=(('stake',3),)
        def _getsizing(self,comminfo,cash,data,isbuy):
            seen.append((self.strategy,cash,comminfo.getcommission(1,10),data,isbuy))
            return -self.p.stake
    engine=bt.Cerebro(stdstats=False);engine.addsizer(Custom,stake=4)
    broker=NautilusBroker(NS(account=lambda:{'cash':125.,'equity':130.}))
    data=NS(_name='A',close=[10.])
    broker.setcommission(commission=.01,percabs=True)
    strategy=NS()
    sizer=native_sizer(engine);sizer.set(strategy,broker)
    assert isinstance(sizer,Custom) and sizer.getsizing(data,False)==-4
    assert seen==[(strategy,125.,.1,data,False)]
    validate_components(engine,account_provider=None)


@pytest.mark.parametrize('kind,expected',[(bt.sizers.PercentSizer,25.),(bt.sizers.AllInSizer,100.)])
def test_builtin_sizers_use_current_native_cash_and_position(kind,expected):
    engine=bt.Cerebro(stdstats=False);engine.addsizer(kind)
    broker=NautilusBroker(NS(account=lambda:{'cash':1000.,'equity':1000.},position=lambda name:bt.Position()))
    data=NS(_name='A',close=[10.]);sizer=native_sizer(engine);sizer.set(NS(),broker)
    # PercentSizer's actual BT/TL default is20%, overridden below to25%.
    if kind is bt.sizers.PercentSizer:sizer.p.percents=25
    assert sizer.getsizing(data,True)==expected


def test_fixed_sizer_preserves_type_params_and_setsizing_without_fake_money():
    engine=bt.Cerebro(stdstats=False);engine.addsizer(bt.sizers.FixedSize,stake=11,tranches=3)
    sizer=native_sizer(engine)
    assert isinstance(sizer,bt.sizers.FixedSize)
    assert sizer.p.stake==11 and sizer.getsizing(None,True)==3
    sizer.p.tranches=1;sizer.setsizing(9)
    assert sizer.getsizing(None,False)==9


def test_sizer_never_fills_missing_native_cash_with_zero():
    engine=bt.Cerebro(stdstats=False);engine.addsizer(bt.sizers.PercentSizer)
    broker=NautilusBroker(NS(account=lambda:{'equity':12.}))
    sizer=native_sizer(engine);sizer.set(NS(),broker)
    with pytest.raises(ValueError,match='cash unavailable'):
        sizer.getsizing(NS(_name='A',close=[10.]),True)
