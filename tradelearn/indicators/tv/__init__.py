"""Qualified TradingView/PyneCore 6.13.3 persistent sources.

Parameter names and input schema are explicit in each callable signature.
Unqualified compatibility composites are no longer dynamically exported.
"""

from .persistent import SPECS as _SPECS
from .persistent import make_indicator as _make_indicator

BACKEND = "pynesys-pynecore==6.13.3"
QUALIFIED = tuple(_SPECS)
for _name in QUALIFIED:
    globals()[_name] = _make_indicator(_name)
    globals()[_name.upper()] = globals()[_name]
__all__ = ["BACKEND", "QUALIFIED"] + [name for item in QUALIFIED for name in (item, item.upper())]
del _name
