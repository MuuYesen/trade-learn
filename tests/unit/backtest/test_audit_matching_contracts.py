from tests.unit.backtest._engine_config import feed as _as_feed
from tests.unit.backtest._engine_config import configured_engine
"""BT orders use their ordered feed; native direct controls retain explicit clocks."""
import pandas as pd
from datetime import timedelta
import pytest

from tradelearn.backtest import engine as engine_module
from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


def _bars(dates):
    return pd.DataFrame(
        dict(open=10., high=11., low=9., close=10., volume=100.),
        index=pd.to_datetime(dates, utc=True),
    )


@pytest.mark.parametrize("runner", ["single", "native-multi", "python-multi"])
@pytest.mark.parametrize("mode", ["exact", "smart"])
@pytest.mark.parametrize("on_close", [False, True])
def test_bt_expiry_waits_for_ordered_feed_even_if_primary_keeps_advancing(monkeypatch, runner, mode, on_close):
    class Expiry(Strategy):
        def init(self):
            self.events = []
            self.states = []

        def next(self):
            if len(self) == 1:
                self.order = self.buy(
                    data=self.datas[-1], size=1, price=1,
                    exectype=Order.Limit, valid=timedelta(days=1),
                )
            self.states.append((self.order.status, self._pending_size.get(self.datas[-1], 0)))

        def notify_order(self, order):
            if order.status == Order.Expired:
                self.events.append(pd.Timestamp(self.data.datetime.datetime(0), tz="UTC"))

    cerebro = configured_engine(match_mode=mode, trade_on_close=on_close)
    cerebro.adddata(_as_feed(_bars(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])), name="clock")
    if runner != "single":
        cerebro.adddata(_as_feed(_bars(["2026-01-01"])), name="target")
    if runner == "python-multi":
        monkeypatch.setattr(engine_module, "_build_clocked_multi_data_runner", lambda datas: None)
    cerebro.addstrategy(Expiry)
    [strategy] = cerebro.run()

    if runner == "single":
        assert strategy.states == [(Order.Accepted, 1), (Order.Accepted, 1),
                                   (Order.Expired, 0), (Order.Expired, 0)]
        assert strategy.events == [pd.Timestamp("2026-01-03", tz="UTC")]
    else:
        # Fixed BT does not advance an inactive order's own expiry clock.
        # The paired ordinary-BT counterexample lives in test_bt_subsecond_clock.
        assert strategy.states == [(Order.Accepted, 1)] * 4
        assert strategy.events == []
    assert strategy.getposition(strategy.datas[-1]).size == 0


@pytest.mark.parametrize('target_clock',[False,True])
def test_native_expiry_clock_is_explicit_and_default_global_policy_stays_unchanged(target_clock):
    from tradelearn import _rust
    native=_rust.RustBacktestEngine([],[],[],[],[],[],1000.,0.,False,False,False,0.,0.,False,False,False)
    ref=native.submit_order_for_symbol('target','buy','limit',.25,1.,None)
    native.configure_order(ref,1,None)
    if target_clock:native.configure_order_expiry_clock(ref,True)
    def step(primary,target):
        native.step_open_bars_compact(['primary','target'],[primary,target],[10.,10.],[10.,10.],[10.,10.],[10.,10.],[1.,1.])
        return native.drain_order_events()
    assert step(0,0)==[]
    assert step(2,0)==([] if target_clock else [(ref,'expired')])
    assert step(3,2)==([(ref,'expired')] if target_clock else [])
    assert native.get_cash()==1000.
