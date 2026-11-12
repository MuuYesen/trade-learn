"""Backtrader-shaped TA classes over the shared qualified persistent source."""
from talib import abstract as _abstract
from tradelearn.indicators import talib as _source
from tradelearn.backtest.indicator import Indicator as _Indicator

__all__ = list(_source.QUALIFIED)


def __getattr__(name):
    if name not in _source.QUALIFIED:
        raise AttributeError(f"BT TA-Lib indicator {name!r} is not qualified")
    function = getattr(_source, name)
    aliases = tuple(_abstract.Function(name).output_names)
    inputs = tuple(function._bar_columns)
    source_aliases = tuple(function.output_names)

    def initialize(self):
        if len(self.datas) != len(inputs):
            raise TypeError(f"{name} requires {len(inputs)} explicit input lines")
        values = tuple(data.lines[0] if hasattr(data, "lines") else
                       data.close if hasattr(data, "close") else data
                       for data in self.datas)
        output = function(*values, **self.p.asdict())
        if len(aliases) == 1:
            setattr(self.lines, aliases[0], output)
        else:
            for alias, source_alias in zip(aliases, source_aliases, strict=True):
                setattr(self.lines, alias, output[source_alias])

    indicator = type(name, (_Indicator,), {
        "__module__": __name__,
        "__doc__": f"{name}: shared qualified TA-Lib state with BT class/line names.",
        "lines": aliases,
        "params": tuple(function.params.items()),
        "__init__": initialize,
    })
    globals()[name] = indicator
    return indicator


def __dir__():
    return sorted(set(globals()) | set(__all__))
