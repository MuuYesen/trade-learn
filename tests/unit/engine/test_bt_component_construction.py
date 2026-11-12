"""Actual BT component construction contracts, not method-name inspection."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl


def engine(api, strategy=None):
    frame = pd.DataFrame(
        {k: [10.0, 11.0] for k in ["open", "high", "low", "close", "volume"]},
        index=pd.date_range("2026-01-01", periods=2),
    )
    e = api.Cerebro(stdstats=False)
    e.addstrategy(strategy or api.Strategy)
    e.adddata(api.feeds.PandasData(dataname=frame))
    return e


def constructor_calls(api):
    calls = []

    class A(api.Analyzer):
        def snapshot(self, stage):
            try:
                length = len(self)
            except TypeError:
                length = "TypeError"
            calls.append(
                (
                    stage,
                    getattr(self, "strategy", None) is not None,
                    getattr(self, "data", None) is not None,
                    getattr(self, "data0", None) is getattr(self, "data", None),
                    getattr(self, "data_close", None)
                    is getattr(getattr(self, "data", None), "close", False),
                    getattr(self, "data0_close", None)
                    is getattr(getattr(self, "data", None), "close", False),
                    getattr(self, "data_0", None)
                    is getattr(getattr(self, "data", None), "close", False),
                    length,
                )
            )

        def create_analysis(self):
            self.rets = {}
            self.snapshot("create_analysis")

        def __init__(self):
            self.snapshot("init")

        def start(self):
            self.snapshot("start")

    e = engine(api)
    e.addanalyzer(A)
    e.run()
    return calls


def test_analyzer_owner_data_aliases_and_length_exist_before_user_constructor():
    assert constructor_calls(tl) == constructor_calls(official)


def observer_params(api, overrides):
    values = []

    class Parent(api.Observer):
        lines = ("observed",)
        params = (("period", 2), ("enabled", True))

        def start(self):
            values.append((self.p.period, getattr(self.p, "enabled", "missing")))

    class Child(Parent):
        params = (("period", 3),)

    e = engine(api)
    e.addobserver(Child, **overrides)
    e.run()
    return values


@pytest.mark.parametrize("overrides", [{}, {"period": 4}, {"enabled": False}])
def test_observer_inherits_parent_params_and_child_defaults(overrides):
    assert observer_params(tl, overrides) == observer_params(official, overrides)


def test_unknown_base_analyzer_argument_is_rejected_like_bt():
    errors = []
    for api in [official, tl]:
        e = engine(api)
        e.addanalyzer(api.Analyzer, not_declared=True)
        try:
            e.run()
            errors.append(None)
        except TypeError:
            errors.append("TypeError")
    assert errors == ["TypeError", "TypeError"]


def analyzer_registration(api, explicit):
    calls = []

    class Named(api.Analyzer):
        params = (("name", "default"),)

        def start(self):
            calls.append(self.p.name)

    e = engine(api)
    for value in ("first", "second"):
        kwargs = {"name": value}
        if explicit:
            kwargs["_name"] = "custom"
        e.addanalyzer(Named, **kwargs)
    strategy = e.run()[0]
    name = "custom" if explicit else "named"
    return calls, tuple(strategy.analyzers.getbyname(key).p.name for key in (name, name + "1"))


@pytest.mark.parametrize("explicit", [False, True])
def test_analyzer_name_parameter_is_forwarded_and_repeated_instances_get_bt_suffix(explicit):
    assert analyzer_registration(tl, explicit) == analyzer_registration(official, explicit)


def test_analyzer_constructor_name_kwarg_is_not_consumed_as_registration_name():
    def calls(api):
        values = []

        class Named(api.Analyzer):
            def __init__(self, name):
                values.append(name)

        e = engine(api)
        e.addanalyzer(Named, name="constructor-value")
        e.run()
        return values

    assert calls(tl) == calls(official)


def observer_constructor_calls(api):
    calls = []

    class Sample(api.Observer):
        lines = ("observed",)
        params = (("period", 2),)

        def snapshot(self, stage):
            calls.append(
                (
                    stage,
                    self.p.period,
                    self._owner is not None,
                    self.data is self.data0,
                    self.data_close is self.data.close,
                    self.data0_close is self.data.close,
                    len(self),
                )
            )

        def __init__(self):
            self.snapshot("init")

        def start(self):
            self.snapshot("start")

    e = engine(api)
    e.addobserver(Sample, period=4)
    e.run()
    return calls


def test_observer_context_and_params_exist_before_custom_constructor_without_super():
    assert observer_constructor_calls(tl) == observer_constructor_calls(official)


def analyzer_collection_calls(api):
    class Sample(api.Analyzer):
        params = (("value", 0),)

    e = engine(api)
    e.addanalyzer(Sample, value=2)
    e.addanalyzer(Sample, value=3)
    collection = e.run()[0].analyzers
    try:
        collection.getbyname("missing")
    except Exception as error:
        missing = type(error).__name__
    return (
        list(collection.getnames()),
        [(name, item.p.value) for name, item in collection.getitems()],
        collection[0].p.value,
        collection[-1].p.value,
        [item.p.value for item in collection],
        missing,
    )


def test_analyzer_collection_names_items_integer_index_and_iteration_match_bt():
    assert analyzer_collection_calls(tl) == analyzer_collection_calls(official)


def test_components_start_before_business_start_like_real_bt():
    def calls(api):
        result = []

        class Business(api.Strategy):
            def start(self):
                result.append("strategy")

        class Analysis(api.Analyzer):
            def start(self):
                result.append("analyzer")

        class Observation(api.Observer):
            lines = ("observed",)

            def start(self):
                result.append("observer")

        e = engine(api, Business)
        e.addanalyzer(Analysis)
        e.addobserver(Observation)
        e.run()
        return result

    assert calls(tl) == calls(official)


def test_observer_positional_args_name_param_and_obsname_match_bt():
    def calls(api):
        values = []
        class Observation(api.Observer):
            lines = ('observed',)
            params = (('name', 'default'),)
            def __init__(self, label):
                values.append((label, self.p.name, self.data is self.data0))
        e = engine(api)
        e.addobserver(Observation, 'label-value', name='user-name', obsname='named')
        strategy = e.run()[0]
        return values, strategy.observers.named.p.name
    assert calls(tl) == calls(official)


def test_analyzer_data_argument_does_not_replace_the_owner_data_context():
    def calls(api):
        values = []
        class Analysis(api.Analyzer):
            def __init__(self, argument):
                values.append((argument is self.strategy.datas[1],
                               self.data is self.strategy.data0,
                               self.data1 is self.strategy.datas[1]))
        e = engine(api)
        frame = pd.DataFrame({key:[20.,21.] for key in ('open','high','low','close','volume')},
                             index=pd.date_range('2026-01-01', periods=2))
        secondary = e.adddata(api.feeds.PandasData(dataname=frame))
        e.addanalyzer(Analysis, secondary)
        e.run()
        return values
    assert calls(tl) == calls(official)


def test_observer_underscore_name_is_a_user_param_not_a_collection_selector():
    def values(api):
        calls = []
        class Observation(api.Observer):
            lines = ('observed',)
            params = (('_name', 'default'),)
            def start(self):
                calls.append(self.p._name)
        e = engine(api)
        e.addobserver(Observation, _name='user-value')
        e.run()
        return calls
    assert values(tl) == values(official)
