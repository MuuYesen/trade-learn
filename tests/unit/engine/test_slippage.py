import pandas as pd
import pytest

from tradelearn.core.costs import BarRangeSlippage, FixedSlippage, PercentSlippage
from tradelearn.engine import Cerebro, Order, Strategy, feeds


class RoundTrip(Strategy):
    params = (("limit", None),)

    def next(self):
        if len(self) == 1:
            if self.p.limit is None:
                self.buy(size=1)
            else:
                self.buy(size=1, exectype=Order.Limit, price=self.p.limit)
        elif len(self) == 2:
            self.sell(size=1)


def _engine(**strategy_options):
    engine = Cerebro(stdstats=False)
    prices = [100.0, 100.0, 110.0]
    bars = pd.DataFrame(
        {"open": prices, "close": prices, "high": [105.0, 105.0, 115.0],
         "low": [95.0, 95.0, 105.0], "volume": 1000.0},
        index=pd.date_range("2024-01-01", periods=3),
    )
    engine.adddata(feeds.PandasData(dataname=bars))
    engine.addstrategy(RoundTrip, **strategy_options)
    engine.broker.setcommission(commission=0.01)
    return engine


@pytest.mark.parametrize(("model", "buy", "sell"), [
    (None, 100.0, 110.0),
    (FixedSlippage(0.5), 100.5, 109.5),
    (PercentSlippage(0.01), 101.0, 108.9),
    (PercentSlippage(0.02), 102.0, 107.8),
])
def test_engine_slippage_changes_fills_and_fees(model, buy, sell):
    engine = _engine()
    engine.broker.configure_matching(slippage=model)
    stats = engine.run()[0].stats
    assert stats.fills["price"].tolist() == pytest.approx([buy, sell])
    assert stats.fills["commission"].tolist() == pytest.approx([buy * 0.01, sell * 0.01])
    if model is not None:
        kind = "fixed" if isinstance(model, FixedSlippage) else "percent"
        assert stats.config["broker"]["slippage"]["kind"] == kind


def test_slippage_respects_limit_price():
    engine = _engine(limit=100.5)
    engine.broker.configure_matching(slippage=PercentSlippage(0.02))
    assert engine.run()[0].stats.fills["price"].iloc[0] == pytest.approx(100.5)


@pytest.mark.parametrize("model", [
    FixedSlippage(-1), FixedSlippage(float("nan")), PercentSlippage(1.0),
    PercentSlippage(float("inf")),
])
def test_invalid_slippage_is_rejected_at_configuration(model):
    with pytest.raises(ValueError, match="slippage"):
        _engine().broker.configure_matching(slippage=model)


def test_unimplemented_slippage_model_is_rejected_at_configuration():
    with pytest.raises(NotImplementedError, match="slippage"):
        _engine().broker.configure_matching(slippage=BarRangeSlippage(0.2))


def test_stale_native_extension_refuses_nonzero_slippage(monkeypatch):
    from types import SimpleNamespace

    from tradelearn.backtest import engine as runtime

    def legacy_engine(*_args):
        return SimpleNamespace(_configure_bt_precision=lambda _preserve: None)

    monkeypatch.setattr(runtime, "_load_rust_backtest_engine", lambda: legacy_engine)
    engine = _engine()
    engine.broker.configure_matching(slippage=FixedSlippage(1.0))
    with pytest.raises(RuntimeError, match="slippage.*native kernel"):
        engine.run()
    assert engine.broker._engine is None


def test_excessive_fixed_slippage_does_not_create_a_negative_fill():
    engine = _engine()
    engine.broker.configure_matching(slippage=FixedSlippage(120.0))
    assert engine.run()[0].stats.fills["price"].tolist() == [220.0]


def test_slippage_cannot_be_changed_after_matching_is_initialized():
    engine = _engine()
    engine.run()
    with pytest.raises(NotImplementedError, match="slippage changes"):
        engine.broker.configure_matching(slippage=PercentSlippage(0.01))
