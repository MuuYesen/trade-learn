"""Backtrader sizing protocol; execution stays in the bound broker."""
from tradelearn.backtest.sizer import Sizer as CoreSizer
from tradelearn.engine.base import MetaParams, Params, collect_param_defaults


class Sizer(CoreSizer, metaclass=MetaParams):
    params = ()

    def _base_init(self, **kwargs):
        self.params = self.p = Params(collect_param_defaults(type(self))[1], **kwargs)

    def __init__(self):
        super().__init__()

    def getsizing(self, data, isbuy):
        return self._getsizing(self.broker.getcommissioninfo(data), self.broker.getcash(), data, isbuy)

    def _getsizing(self, comminfo, cash, data, isbuy):
        raise NotImplementedError('Sizer subclasses must implement _getsizing')


class FixedSize(Sizer):
    params = (("stake", 1), ("tranches", 1))

    def _getsizing(self, comminfo, cash, data, isbuy):
        return abs(int(self.p.stake/self.p.tranches)) if self.p.tranches > 1 else self.p.stake

    def setsizing(self, stake):
        self.p.stake = abs(int(self.p.stake/self.p.tranches)) if self.p.tranches > 1 else stake


class PercentSizer(Sizer):
    params = (("percents", 20), ("retint", False))

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        size = position.size if position else cash/data.close[0]*self.p.percents/100
        return int(size) if self.p.retint else size


class AllInSizer(PercentSizer):
    params = (("percents", 100),)


__all__ = ["AllInSizer", "FixedSize", "PercentSizer", "Sizer"]
