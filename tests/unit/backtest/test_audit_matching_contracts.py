"""Orders expire on the runtime clock, including while their feed is inactive."""
import pandas as pd
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
def test_expiry_uses_primary_clock_without_a_target_bar(monkeypatch, runner, mode, on_close):
    class Expiry(Strategy):
        def init(self):
            self.events = []
            self.states = []

        def next(self):
            if len(self) == 1:
                self.order = self.buy(
                    data=self.datas[-1], size=1, price=1,
                    exectype=Order.Limit, valid=86400,
                )
            self.states.append((self.order.status, self._pending_size.get(self.datas[-1], 0)))

        def notify_order(self, order):
            if order.status == Order.Expired:
                self.events.append(self.data.datetime[0])

    cerebro = Cerebro(match_mode=mode, trade_on_close=on_close)
    cerebro.adddata(_bars(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]), name="clock")
    if runner != "single":
        cerebro.adddata(_bars(["2026-01-01"]), name="target")
    if runner == "python-multi":
        monkeypatch.setattr(engine_module, "_build_clocked_multi_data_runner", lambda datas: None)
    cerebro.addstrategy(Expiry)
    [strategy] = cerebro.run()

    assert strategy.states == [(Order.Accepted, 1), (Order.Accepted, 1),
                               (Order.Expired, 0), (Order.Expired, 0)]
    assert strategy.events == [pd.Timestamp("2026-01-03", tz="UTC")]
    assert strategy.getposition(strategy.datas[-1]).size == 0
