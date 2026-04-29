"""Signal-driven strategy support for the engine facade.

Mirrors backtrader's signal API: users declare indicator-based signals,
and ``SignalStrategy`` auto-generates buy/sell/close orders each bar.
"""

from __future__ import annotations

import collections
from typing import Any

from tradelearn.engine.indicators import Indicator
from tradelearn.engine.strategy import Strategy

SIGNAL_NONE = 0
SIGNAL_LONGSHORT = 1
SIGNAL_LONG = 2
SIGNAL_LONG_INV = 3
SIGNAL_LONG_ANY = 4
SIGNAL_SHORT = 5
SIGNAL_SHORT_INV = 6
SIGNAL_SHORT_ANY = 7
SIGNAL_LONGEXIT = 8
SIGNAL_LONGEXIT_INV = 9
SIGNAL_LONGEXIT_ANY = 10
SIGNAL_SHORTEXIT = 11
SIGNAL_SHORTEXIT_INV = 12
SIGNAL_SHORTEXIT_ANY = 13


class Signal(Indicator):
    """Wraps any single-line indicator as a signal source."""

    def __init__(self, data: Any) -> None:
        self._signal_data = data

    @property
    def signal_value(self) -> float:
        try:
            return float(self._signal_data[0])
        except (IndexError, TypeError):
            return 0.0

    def __getitem__(self, index: int) -> float:
        try:
            return float(self._signal_data[index])
        except (IndexError, TypeError):
            return 0.0


class SignalStrategy(Strategy):
    """Strategy subclass that auto-trades based on registered signals.

    Signals are indicator lines whose current value (``> 0`` or ``< 0``)
    drives entry/exit decisions.  Users can still define ``next()`` for
    custom logic executed *after* signal processing.

    Usage via ``cerebro.add_signal``::

        cerebro.add_signal(SIGNAL_LONG, MySignalIndicator, period=20)

    Or directly inside ``__init__``::

        class MySigStrategy(SignalStrategy):
            def __init__(self):
                sma = self.data.close.ta.sma(20)
                sig = Signal(sma - self.data.close)
                self.signal_add(SIGNAL_LONGSHORT, sig)
    """

    _accumulate: bool = False
    _concurrent: bool = False
    params = (
        ("signals", []),
        ("_accumulate", False),
        ("_concurrent", False),
        ("_data", None),
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        user_next = cls.__dict__.get("next")
        if user_next is None or user_next is SignalStrategy.next:
            return
        if "_next_custom" not in cls.__dict__:
            cls._next_custom = user_next
        cls.next = SignalStrategy.next

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        signals_spec = getattr(self.p, "signals", [])
        self._accumulate = bool(getattr(self.p, "_accumulate", False))
        self._concurrent = bool(getattr(self.p, "_concurrent", False))
        _data_target = getattr(self.p, "_data", None)

        self._signals: dict[int, list[Any]] = collections.defaultdict(list)
        self._sentinel: Any = None

        if _data_target is None:
            self._dtarget = self.datas[0] if self.datas else None
        elif isinstance(_data_target, int):
            self._dtarget = self.datas[_data_target]
        elif isinstance(_data_target, str):
            self._dtarget = self.getdatabyname(_data_target)
        else:
            self._dtarget = _data_target

        for sigtype, sigcls, sigargs, sigkwargs in signals_spec:
            self._signals[sigtype].append(sigcls(*sigargs, **sigkwargs))

        self._has_longshort = False
        self._has_long = False
        self._has_short = False
        self._has_longexit = False
        self._has_shortexit = False

    def start(self) -> None:
        super().start()
        self._has_longshort = bool(self._signals[SIGNAL_LONGSHORT])
        self._has_long = bool(self._signals[SIGNAL_LONG])
        self._has_short = bool(self._signals[SIGNAL_SHORT])
        self._has_longexit = bool(self._signals[SIGNAL_LONGEXIT])
        self._has_shortexit = bool(self._signals[SIGNAL_SHORTEXIT])

    def signal_add(self, sigtype: int, signal: Any) -> None:
        self._signals[sigtype].append(signal)

    def next(self) -> None:
        self._next_signal()
        self._next_custom()

    def _next_custom(self) -> None:
        """Override point for user logic after signal processing."""

    def notify_order(self, order: Any) -> None:
        if self._sentinel is not None and order is self._sentinel:
            if not getattr(order, "alive", lambda: True)():
                self._sentinel = None
        super().notify_order(order)

    def _sig_val(self, sig: Any) -> float:
        if isinstance(sig, Signal):
            return sig.signal_value
        try:
            return float(sig[0])
        except (IndexError, TypeError):
            return 0.0

    def _next_signal(self) -> None:
        if self._sentinel is not None and not self._concurrent:
            return

        sigs = self._signals
        _v = self._sig_val

        def _all_pos(lst: list[Any]) -> bool:
            return bool(lst) and all(_v(x) > 0.0 for x in lst)

        def _all_neg(lst: list[Any]) -> bool:
            return bool(lst) and all(_v(x) < 0.0 for x in lst)

        def _all_nonzero(lst: list[Any]) -> bool:
            return bool(lst) and all(_v(x) != 0.0 for x in lst)

        ls_long = _all_pos(sigs[SIGNAL_LONGSHORT])
        ls_short = _all_neg(sigs[SIGNAL_LONGSHORT])

        l_enter = (
            _all_pos(sigs[SIGNAL_LONG])
            or _all_neg(sigs[SIGNAL_LONG_INV])
            or _all_nonzero(sigs[SIGNAL_LONG_ANY])
        )
        s_enter = (
            _all_neg(sigs[SIGNAL_SHORT])
            or _all_pos(sigs[SIGNAL_SHORT_INV])
            or _all_nonzero(sigs[SIGNAL_SHORT_ANY])
        )

        l_exit = (
            _all_neg(sigs[SIGNAL_LONGEXIT])
            or _all_pos(sigs[SIGNAL_LONGEXIT_INV])
            or _all_nonzero(sigs[SIGNAL_LONGEXIT_ANY])
        )
        s_exit = (
            _all_pos(sigs[SIGNAL_SHORTEXIT])
            or _all_neg(sigs[SIGNAL_SHORTEXIT_INV])
            or _all_nonzero(sigs[SIGNAL_SHORTEXIT_ANY])
        )

        l_rev = not self._has_longexit and s_enter
        s_rev = not self._has_shortexit and l_enter

        l_leave = not self._has_longexit and (
            _all_neg(sigs[SIGNAL_LONG])
            or _all_pos(sigs[SIGNAL_LONG_INV])
            or _all_nonzero(sigs[SIGNAL_LONG_ANY])
        )
        s_leave = not self._has_shortexit and (
            _all_pos(sigs[SIGNAL_SHORT])
            or _all_neg(sigs[SIGNAL_SHORT_INV])
            or _all_nonzero(sigs[SIGNAL_SHORT_ANY])
        )

        size = self.getposition(self._dtarget).size if self._dtarget else 0

        if not size:
            if ls_long or l_enter:
                self._sentinel = self.buy(data=self._dtarget)
            elif ls_short or s_enter:
                self._sentinel = self.sell(data=self._dtarget)

        elif size > 0:
            if ls_short or l_exit or l_rev or l_leave:
                self.close(data=self._dtarget)
            if ls_short or l_rev:
                self._sentinel = self.sell(data=self._dtarget)
            if (ls_long or l_enter) and self._accumulate:
                self._sentinel = self.buy(data=self._dtarget)

        elif size < 0:
            if ls_long or s_exit or s_rev or s_leave:
                self.close(data=self._dtarget)
            if ls_long or s_rev:
                self._sentinel = self.buy(data=self._dtarget)
            if (ls_short or s_enter) and self._accumulate:
                self._sentinel = self.sell(data=self._dtarget)
