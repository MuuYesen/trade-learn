"""Frozen Backtrader commission formulas; execution qualification is separate."""
from tradelearn.backtest.parameters import Params,collect_param_defaults,split_param_kwargs


class _MetaCommission(type):
    def __call__(cls,*args,**kwargs):
        parameters,other=split_param_kwargs(cls,kwargs)
        instance=cls.__new__(cls)
        instance._base_init(**parameters)
        instance.__init__(*args,**other)
        return instance


class CommInfoBase(metaclass=_MetaCommission):
    """Backtrader commission/margin equations with percentage defaults."""
    COMM_PERC, COMM_FIXED = range(2)
    params = (("commission", 0.0), ("mult", 1.0), ("margin", None),
              ("commtype", None), ("stocklike", False), ("percabs", False),
              ("interest", 0.0), ("interest_long", False), ("leverage", 1.0),
              ("automargin", False))

    def _base_init(self, **kwargs):
        self.p = self.params = Params(collect_param_defaults(type(self))[1], **kwargs)
        self._stocklike = self.p.stocklike
        self._commtype = self.p.commtype
        if self._commtype is None:
            self._stocklike = not bool(self.p.margin)
            self._commtype = self.COMM_PERC if self._stocklike else self.COMM_FIXED
        if not self._stocklike and not self.p.margin:
            self.p.margin = 1.0
        if self._commtype == self.COMM_PERC and not self.p.percabs:
            self.p.commission /= 100
        self._creditrate = self.p.interest / 365

    def __init__(self):
        pass

    @property
    def margin(self): return self.p.margin
    @property
    def stocklike(self): return self._stocklike

    def get_margin(self, price):
        if not self.p.automargin: return self.p.margin
        return price * (self.p.mult if self.p.automargin < 0 else self.p.automargin)

    def get_leverage(self): return self.p.leverage

    def getsize(self, price, cash):
        return int(self.p.leverage * (cash // (price if self._stocklike else self.get_margin(price))))

    def getoperationcost(self, size, price):
        return abs(size) * (price if self._stocklike else self.get_margin(price))

    def getvaluesize(self, size, price):
        return size * price if self._stocklike else abs(size) * self.get_margin(price)

    def getvalue(self, position, price):
        if not self._stocklike: return abs(position.size) * self.get_margin(price)
        if position.size >= 0: return position.size * price
        return position.size * (2 * position.price - price)

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission * (price if self._commtype == self.COMM_PERC else 1)

    def getcommission(self, size, price): return self._getcommission(size, price, True)
    def confirmexec(self, size, price): return self._getcommission(size, price, False)

    def profitandloss(self, size, price, newprice):
        return size * (newprice-price) * self.p.mult

    def cashadjust(self, size, price, newprice):
        return 0.0 if self._stocklike else self.profitandloss(size,price,newprice)

    def get_credit_interest(self, data, pos, dt):
        if pos.size > 0 and not self.p.interest_long: return 0.0
        days = (dt.date()-pos.datetime.date()).days
        return 0.0 if days <= 0 else self._get_credit_interest(data,pos.size,pos.price,days,dt.date(),pos.datetime.date())

    def _get_credit_interest(self, data, size, price, days, dt0, dt1):
        return days * self._creditrate * abs(size) * price


class CommissionInfo(CommInfoBase):
    params = (("percabs", True),)


__all__ = ["CommInfoBase", "CommissionInfo"]
