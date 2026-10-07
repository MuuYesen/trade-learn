"""Bounded, JSON-safe Charting catalog: 104 studies plus Smart Money Concepts.

Input bars must be chronological, unique, UTC indexed OHLCV. Series event
studies emit at confirmation time. SMC returns snapshot geometry with explicit
pivot confirmation metadata. Profiles distribute OHLCV uniformly across each bar.
"""

import json
import math
import threading
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .formulas import compute
from .special import profile

_CATALOG = json.loads(Path(__file__).with_name("catalog.json").read_text())
_LOOKUP = {key: entry for entry in _CATALOG for key in (entry["id"], entry["name"])}
_LOCK = threading.RLock()  # PyneCore resets function isolation between calls.


def catalog():
    return deepcopy(_CATALOG)


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame) or not {"open", "high", "low", "close", "volume"} <= set(
        frame.columns
    ):
        raise ValueError("OHLCV DataFrame required")
    if len(frame) > 10000:
        raise ValueError("At most 10000 bars per calculation")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Timezone-aware DatetimeIndex required")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Bar timestamps must be unique and increasing")
    f = frame[["open", "high", "low", "close", "volume"]].astype(float).copy()
    f.index = f.index.tz_convert("UTC")
    if not np.isfinite(f.to_numpy()).all():
        raise ValueError("Bars must contain finite numbers")
    if (
        (f.volume < 0).any()
        or (f.high < f.low).any()
        or (f.high < f[["open", "close"]].max(axis=1)).any()
        or (f.low > f[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("Invalid OHLCV bounds")
    return f


def calculate(frame, name_or_id, params=None, secondary=None, visible_range=None):
    entry = _LOOKUP.get(name_or_id)
    if entry is None:
        raise ValueError("Unknown indicator: " + str(name_or_id))
    params = params or {}
    definitions = {p["id"]: p for p in entry["params"]}
    if not isinstance(params, dict) or set(params) - set(definitions):
        raise ValueError("Unknown indicator parameter")
    p = {k: d["default"] for k, d in definitions.items()}
    p.update(params)
    for k, d in definitions.items():
        x = p[k]
        kind = d["type"]
        if kind in ["integer", "number"]:
            if (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or (kind == "integer" and int(x) != x)
            ):
                raise ValueError("Invalid " + k)
            if x < d.get("min", -math.inf) or x > d.get("max", math.inf):
                raise ValueError("Out of range " + k)
            if kind == "integer":
                p[k] = int(x)
        elif kind == "select" and x not in d["options"]:
            raise ValueError("Invalid " + k)
        elif kind == "string" and (not isinstance(x, str) or len(x) > 120):
            raise ValueError("Invalid " + k)
    f = _validate_frame(frame)
    if "symbol" in p:
        if secondary is None:
            raise ValueError("This indicator requires secondary OHLCV bars")
        secondary = _validate_frame(secondary).reindex(f.index)
    name = entry["name"]
    notes = list(entry.get("notes", []))
    if name == "Smart Money Concepts":
        from .smc import drawings

        return dict(series=[], drawings=drawings(f, p["depth"]), notes=notes)
    if name.startswith("Volume Profile"):
        return dict(
            series=[],
            profile=profile(f, p, visible_range if name.endswith("Visible Range") else None),
            notes=[
                "OHLCV approximation: each bar volume is distributed uniformly "
                "over its high-low range; not a tick-level volume profile."
            ],
        )
    if not len(f):
        return dict(series=[dict(id=d["id"], values=[]) for d in entry["plots"]], notes=notes)
    with _LOCK:
        result = compute(f, name, p, secondary)
    if set(result) != set(d["id"] for d in entry["plots"]):
        raise RuntimeError("Output schema mismatch: " + name)
    series = []
    for d in entry["plots"]:
        values = result[d["id"]]
        if len(values) != len(f):
            raise RuntimeError("Output length mismatch")
        series.append(
            dict(
                id=d["id"],
                values=[float(x) if pd.notna(x) and np.isfinite(x) else None for x in values],
            )
        )
    return dict(series=series, notes=notes)
