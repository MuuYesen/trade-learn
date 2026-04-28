from __future__ import annotations

import pandas as pd

import tradelearn as bt


def _ohlcv(rows: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 09:30", periods=rows, freq="1min")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000.0] * rows,
        },
        index=idx,
    )


def test_backtrader_facade_exports_common_surface() -> None:
    for name in [
        "Cerebro",
        "DataFeed",
        "Strategy",
        "Order",
        "TimeFrame",
        "FixedSize",
        "PercentSizer",
        "AllInSizer",
        "CommInfoBase",
        "observers",
        "sizers",
        "feeds",
        "analyzers",
    ]:
        assert hasattr(bt, name)


def test_cerebro_exposes_runtime_compat_methods() -> None:
    cerebro = bt.Cerebro()
    for name in [
        "addsizer",
        "setsizer",
        "resampledata",
        "replaydata",
        "addobserver",
        "addwriter",
        "optstrategy",
        "runstop",
        "plot",
        "set_coc",
        "addtimer",
        "addcalendar",
    ]:
        assert callable(getattr(cerebro, name))


def test_resampledata_adds_higher_timeframe_feed() -> None:
    cerebro = bt.Cerebro()
    data0 = bt.DataFeed(_ohlcv(15), name="1m")
    cerebro.adddata(data0)

    data1 = cerebro.resampledata(data0, timeframe=bt.TimeFrame.Minutes, compression=5)

    assert data1._name == "1m_5Minutes"
    assert len(cerebro.datas) == 2
    assert data1.buflen() < data0.buflen()


def test_sizer_observer_and_analyzer_attribute_access() -> None:
    class BuyOnce(bt.Strategy):
        def next(self) -> None:
            if len(self) == 1:
                self.buy()

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.addstrategy(BuyOnce)
    cerebro.addsizer(bt.FixedSize, stake=7)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.Returns)

    strategy = cerebro.run()[0]

    assert abs(strategy.broker.fills_frame().iloc[0]["size"]) == 7
    assert "value" in strategy.observers
    assert strategy.observers.value.get_analysis()["value"]
    assert strategy.analyzers.returns.get_analysis()


def test_bracket_order_helpers_preserve_parent_and_oco_metadata() -> None:
    class BracketStrategy(bt.Strategy):
        def next(self) -> None:
            if len(self) == 1:
                self.orders = self.buy_bracket(
                    size=1,
                    price=101.0,
                    stopprice=98.0,
                    limitprice=105.0,
                )

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.addstrategy(BracketStrategy)
    strategy = cerebro.run()[0]

    main, stop, limit = strategy.orders
    assert main.parent is None
    assert stop.parent is main
    assert limit.parent is main
    assert limit.oco is stop
    assert stop.exectype == bt.Order.Stop
    assert limit.exectype == bt.Order.Limit


def test_commission_scheme_methods_and_datetime_helpers() -> None:
    comminfo = bt.CommInfoBase(commission=0.001, mult=10.0)
    assert comminfo.getcommission(size=2, price=100) == 2.0
    assert comminfo.profitandloss(size=2, price=100, newprice=105) == 100.0

    dt = pd.Timestamp("2024-01-01 09:30")
    assert bt.num2date(bt.date2num(dt)).replace(tzinfo=None) == dt.to_pydatetime()
