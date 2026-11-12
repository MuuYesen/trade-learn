"""Qualified persistent TA-Lib source (requires exactly 0.8.3).

Display-shifted/repainting outputs stay internal until confirmation-time
semantics are separately qualified. No arbitrary upstream fallback export.
"""

from talib import get_functions as _get_functions

from .persistent import make_indicator as _make_indicator
from .persistent import require_backend as _require_backend

_require_backend()
EXCLUDED = {
    "DPO": "display-shifted output; confirmation-time API not qualified",
    "FRACTAL": "display-shifted pattern; confirmation-time API not qualified",
    "ICHIMOKU": "display-shifted outputs; confirmation-time API not qualified",
    "ZIGZAG": "pivot/revision semantics not qualified",
}
QUALIFIED = tuple(name for name in _get_functions() if name not in EXCLUDED)
BACKEND = "TA-Lib==0.8.3"
for _name in QUALIFIED:
    globals()[_name] = _make_indicator(_name)
    globals()[_name.lower()] = globals()[_name]
__all__ = ["BACKEND", "QUALIFIED"] + [name for item in QUALIFIED for name in (item, item.lower())]
del _name
