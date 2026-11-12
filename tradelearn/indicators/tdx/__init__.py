"""Qualified persistent indicators from the fixed MyTT upstream formula set."""

from .upstream import SUPPORTED as QUALIFIED
from .upstream import make_indicator as _make_indicator

BACKEND = "MyTT@7cd36ae13cae56657284badc8d5a6b7b8ed62a37"
for _name in QUALIFIED:
    globals()[_name] = _make_indicator(_name)
    globals()[_name.lower()] = globals()[_name]
__all__ = ["BACKEND", "QUALIFIED"] + [name for item in QUALIFIED for name in (item, item.lower())]
del _name
