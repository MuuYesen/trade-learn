"""Private completed-analysis cache primitives; no public research DSL."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd


def analysis_key(values, config, *, version=1):
    digest = hashlib.sha256(repr((version, config)).encode())
    for value in values:
        if value is None:
            digest.update(b"none")
            continue
        digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
        digest.update(
            repr(
                (
                    value.index.names,
                    getattr(value, "name", None),
                    getattr(value, "columns", None),
                    value.dtypes,
                )
            ).encode()
        )
        digest.update(json.dumps(value.attrs, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def copy_analysis(value):
    return {
        **deepcopy({key: item for key, item in value.items() if key != "figures"}),
        "figures": dict(value.get("figures", {})),
    }


def wants_figures(path, format=None):
    return (format or Path(path).suffix.lstrip(".") or "html").lower() in {"html", "htm"}
