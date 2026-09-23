"""Custom indicators compare the current value, while registration uses identity."""
import operator

import pytest

from tradelearn.backtest.lines import LineSeries
from tradelearn.engine import Indicator, Strategy


class Sample(Indicator):
    lines = ("value",)

    def __init__(self, values):
        self.lines.value = LineSeries(values)


@pytest.mark.parametrize("compare", [operator.eq, operator.ne, operator.le, operator.ge,
                                    operator.lt, operator.gt])
def test_custom_indicator_compares_current_value(compare):
    indicator = Sample([1.0, -1.0, 0.0])
    other = Sample([1.0, 0.0, 0.0])
    for index, value in enumerate([1.0, -1.0, 0.0]):
        indicator._advance(index)
        other._advance(index)
        assert bool(compare(indicator, 1.0)) == compare(value, 1.0)
        assert bool(compare(1.0, indicator)) == compare(1.0, value)
        assert bool(compare(indicator, other)) == compare(value, other[0])


def test_indicator_registration_keeps_distinct_equal_valued_lines():
    class Registry:
        _register_indicator = Strategy._register_indicator

        def __init__(self):
            self._indicators = []

    registry = Registry()
    first, second = LineSeries([1.0]), LineSeries([1.0])
    registry._register_indicator(first)
    registry._register_indicator(second)
    registry._register_indicator(first)
    assert len(registry._indicators) == 2
    assert registry._indicators[0] is first
    assert registry._indicators[1] is second
