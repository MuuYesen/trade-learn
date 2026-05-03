from __future__ import annotations

import pandas as pd

import tradelearn as tl
import tradelearn.engine as bt


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
        "OptReturn",
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


def test_root_namespace_does_not_proxy_engine_strategy_surface() -> None:
    for name in ("Cerebro", "Strategy", "Order", "TimeFrame", "feeds", "analyzers"):
        assert not hasattr(tl, name)


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
        "optcallback",
        "runstop",
        "plot",
        "set_coc",
        "addtimer",
        "addcalendar",
    ]:
        assert callable(getattr(cerebro, name))


def test_set_coc_keeps_cheat_on_close_out_of_generic_kwargs() -> None:
    cerebro = bt.Cerebro()

    cerebro.set_coc(True)

    assert cerebro.trade_on_close is True
    assert cerebro.broker._trade_on_close is True
    assert "cheat_on_close" not in cerebro.kwargs


def test_broker_exposes_backtrader_cash_value_aliases() -> None:
    cerebro = bt.Cerebro()

    cerebro.broker.set_cash(12345.0)

    assert cerebro.broker.get_cash() == 12345.0
    assert cerebro.broker.getcash() == 12345.0
    assert cerebro.broker.get_value() == 12345.0
    assert cerebro.broker.getvalue() == 12345.0


def test_cerebro_getbroker_and_broker_addcommissioninfo() -> None:
    cerebro = bt.Cerebro()
    comminfo = bt.CommInfoBase(commission=0.002, mult=5.0)

    assert cerebro.getbroker() is cerebro.broker
    cerebro.broker.addcommissioninfo(comminfo)

    assert cerebro.broker.getcommissioninfo(None) is comminfo
    assert cerebro.broker.commission_ratio == 0.002
    assert cerebro.broker.get_mult() == 5.0


def test_cerebro_setbroker_replaces_broker_object() -> None:
    cerebro = bt.Cerebro()
    broker = cerebro.broker.__class__(cash=4321.0)

    cerebro.setbroker(broker)

    assert cerebro.broker is broker
    assert cerebro.getbroker() is broker


def test_cerebro_addstore_keeps_store_reference() -> None:
    cerebro = bt.Cerebro()
    store = object()

    cerebro.addstore(store)

    assert cerebro.stores == [store]


def test_cerebro_writer_query_surface_aggregates_registered_writers() -> None:
    class Writer:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def getheaders(self) -> list[str]:
            return [f"{self.prefix}-header"]

        def getinfo(self) -> dict[str, str]:
            return {"writer": self.prefix}

        def getvalues(self) -> list[str]:
            return [f"{self.prefix}-value"]

    cerebro = bt.Cerebro()
    assert cerebro.getwriterheaders() == []
    assert cerebro.getwriterinfo() == []
    assert cerebro.getwritervalues() == []

    cerebro.addwriter(Writer, "run")

    assert cerebro.getwriterheaders() == ["run-header"]
    assert cerebro.getwriterinfo() == [{"writer": "run"}]
    assert cerebro.getwritervalues() == ["run-value"]


def test_cerebro_run_accepts_backtrader_runtime_kwargs() -> None:
    class CountBars(bt.Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def next(self) -> None:
            self.calls += 1

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(4))
    cerebro.addstrategy(CountBars)

    [strategy] = cerebro.run(
        runonce=False,
        preload=False,
        maxcpus=1,
        optreturn=False,
        exactbars=True,
        stdstats=False,
        cheat_on_close=True,
    )

    assert strategy.calls == 4
    assert cerebro.exactbars is True
    assert cerebro.stdstats is False
    assert cerebro.trade_on_close is True
    assert cerebro.broker._trade_on_close is True


def test_engine_adddata_normalizes_extra_columns_to_lowercase() -> None:
    seen: dict[str, float] = {}

    class FactorStrategy(bt.Strategy):
        def next(self) -> None:
            if len(self.data) == 3:
                seen["factor"] = self.data.get_array("factor")[self.data._cursor]

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(5).assign(FACTOR=[1.0, 2.0, 3.0, 4.0, 5.0]))
    cerebro.addstrategy(FactorStrategy)

    cerebro.run()

    assert seen == {"factor": 3.0}


def test_addanalyzer_preserves_positional_arguments() -> None:
    class NamedAnalyzer(bt.Analyzer):
        def __init__(self, prefix: str, suffix: str = "") -> None:
            self.prefix = prefix
            self.suffix = suffix

        def get_analysis(self) -> dict[str, str]:
            return {"name": f"{self.prefix}{self.suffix}"}

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(3))
    cerebro.addstrategy(bt.Strategy)
    cerebro.addanalyzer(NamedAnalyzer, "run", suffix="-ok", _name="named")

    [strategy] = cerebro.run()

    assert strategy.analyzers.named.get_analysis() == {"name": "run-ok"}
    assert strategy.stats.analyzers["named"] == {"name": "run-ok"}


def test_broker_order_history_and_open_orders_surface() -> None:
    class SubmitOnce(bt.Strategy):
        def next(self) -> None:
            if len(self) == 1:
                self.order = self.buy(size=1)

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(4))
    cerebro.addstrategy(SubmitOnce)
    strategy = cerebro.run()[0]

    history = strategy.broker.get_orders_history()
    assert strategy.order in history
    assert strategy.broker.get_orders_open() == []


def test_cerebro_tracks_datas_by_name() -> None:
    cerebro = bt.Cerebro()
    data = cerebro.adddata(_ohlcv(), name="asset0")

    assert cerebro.datasbyname["asset0"] is data


def test_cerebro_chaindata_and_rolloverdata_register_feeds() -> None:
    cerebro = bt.Cerebro()
    first = bt.DataFeed(_ohlcv(4), name="first")
    second = bt.DataFeed(_ohlcv(5), name="second")

    assert cerebro.chaindata(first, second) is first
    rolled = cerebro.rolloverdata(_ohlcv(6), name="rolled")

    assert cerebro.datas == [first, second, rolled]
    assert cerebro.datasbyname["first"] is first
    assert cerebro.datasbyname["second"] is second
    assert cerebro.datasbyname["rolled"] is rolled


def test_strategy_getpositionbyname_resolves_named_data() -> None:
    class BuyNamed(bt.Strategy):
        def next(self) -> None:
            if len(self) == 1:
                self.buy(data=self.getdatabyname("asset0"), size=1)
            self.named_position = self.getpositionbyname("asset0")

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(4), name="asset0")
    cerebro.addstrategy(BuyNamed)
    strategy = cerebro.run()[0]

    assert strategy.named_position.size == 1


def test_resampledata_adds_higher_timeframe_feed() -> None:
    cerebro = bt.Cerebro()
    data0 = bt.DataFeed(_ohlcv(15), name="1m")
    cerebro.adddata(data0)

    data1 = cerebro.resampledata(data0, timeframe=bt.TimeFrame.Minutes, compression=5)

    assert data1._name == "1m_5Minutes"
    assert len(cerebro.datas) == 2
    assert data1.buflen() < data0.buflen()


def test_optstrategy_runs_parameter_grid() -> None:
    class ParamStrategy(bt.Strategy):
        params = (("fast", 1), ("slow", 2))

        def next(self) -> None:
            pass

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.optstrategy(ParamStrategy, fast=[1, 2], slow=[3, 4])

    results = cerebro.run()

    assert [(strategy.p.fast, strategy.p.slow) for strategy in results] == [
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]


def test_optstrategy_can_return_backtrader_style_optreturn() -> None:
    class ParamStrategy(bt.Strategy):
        params = (("fast", 1),)

        def next(self) -> None:
            pass

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.optstrategy(ParamStrategy, fast=[1, 2])

    results = cerebro.run(optreturn=True)

    assert [result.p.fast for result in results] == [1, 2]
    assert [result.params.fast for result in results] == [1, 2]
    assert [result.strategycls for result in results] == [ParamStrategy, ParamStrategy]
    assert all(result.analyzers is not None for result in results)


def test_optcallback_receives_full_strategy_instances() -> None:
    class ParamStrategy(bt.Strategy):
        params = (("fast", 1),)

        def next(self) -> None:
            pass

    seen: list[int] = []
    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.optstrategy(ParamStrategy, fast=[3, 5])
    cerebro.optcallback(lambda strategy: seen.append(strategy.p.fast))

    results = cerebro.run()

    assert [strategy.p.fast for strategy in results] == [3, 5]
    assert seen == [3, 5]


def test_runstop_stops_later_strategy_callbacks() -> None:
    class StopAfterTwo(bt.Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def next(self) -> None:
            self.calls += 1
            if self.calls == 2:
                self.cerebro.runstop()

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(10))
    cerebro.addstrategy(StopAfterTwo)

    [strategy] = cerebro.run()

    assert strategy.calls == 2


def test_runstop_state_is_reset_between_runs() -> None:
    class StopImmediately(bt.Strategy):
        def next(self) -> None:
            self.cerebro.runstop()

    class CountAll(bt.Strategy):
        def __init__(self) -> None:
            self.calls = 0

        def next(self) -> None:
            self.calls += 1

    first = bt.Cerebro()
    first.adddata(_ohlcv(5))
    first.addstrategy(StopImmediately)
    first.run()

    second = bt.Cerebro()
    second.adddata(_ohlcv(5))
    second.addstrategy(CountAll)

    [strategy] = second.run()

    assert strategy.calls == 5


def test_addminperiod_extends_warmup_without_backtest_public_api() -> None:
    class WarmupStrategy(bt.Strategy):
        def __init__(self) -> None:
            self.addminperiod(4)
            self.calls: list[int] = []

        def next(self) -> None:
            self.calls.append(len(self))

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv(6))
    cerebro.addstrategy(WarmupStrategy)

    [strategy] = cerebro.run()

    assert strategy.calls == [4, 5, 6]


def test_sizer_observer_and_analyzer_attribute_access() -> None:
    class BuyOnce(bt.Strategy):
        def next(self) -> None:
            if len(self) == 1:
                self.bound_sizer = self.getsizer()
                self.buy()

    cerebro = bt.Cerebro()
    cerebro.adddata(_ohlcv())
    cerebro.addstrategy(BuyOnce)
    cerebro.addsizer(bt.FixedSize, stake=7)
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.Returns)

    strategy = cerebro.run()[0]

    assert abs(strategy.broker.fills_frame().iloc[0]["size"]) == 7
    assert isinstance(strategy.bound_sizer, bt.FixedSize)
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
