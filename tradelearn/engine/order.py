"""BT constants on the shared order projection, without another order ledger."""

from datetime import timedelta

from tradelearn.backtest.models import Order as _CoreOrder


class _OrderType(type):
    def __instancecheck__(cls, value):
        return isinstance(value, _CoreOrder)

    def __subclasscheck__(cls, value):
        return issubclass(value, _CoreOrder)

    def __getattribute__(cls, name):
        if name in {'GTC', 'IOC'}:
            raise AttributeError(name)
        return super().__getattribute__(name)

    def __dir__(cls):
        return [name for name in super().__dir__() if name not in {'GTC', 'IOC'}]


class Order(_CoreOrder, metaclass=_OrderType):
    """Backtrader Order view; callbacks still carry the common core object.

    GTC/IOC are native broker kwargs, not Backtrader Order constants. The Lite
    core keeps its string TIF constants, while Engine DAY is BT's duration0.
    """

    DAY = timedelta(0)
