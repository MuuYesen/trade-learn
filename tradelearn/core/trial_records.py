"""Shared trial evidence and input fingerprints for historical research."""

from contextvars import ContextVar


def _describe(value, seen=None):
    """Stable JSON configuration facts; never serialize a runtime or its repr."""
    import dataclasses
    import datetime
    import hashlib
    import math
    from decimal import Decimal

    import numpy as np
    import pandas as pd

    seen = set() if seen is None else seen
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (Decimal, datetime.timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return _describe(value.item(), seen)
    if isinstance(value, (pd.DataFrame, pd.Series)):
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(value, index=True).values.tobytes()
        ).hexdigest()
        return {
            "sha256": digest,
            "rows": len(value),
            "columns": list(map(str, value.columns))
            if isinstance(value, pd.DataFrame)
            else [str(value.name)],
            "dtypes": list(map(str, value.dtypes))
            if isinstance(value, pd.DataFrame)
            else [str(value.dtype)],
            "index_names": list(value.index.names),
            "attrs": _describe(value.attrs, seen),
        }
    if isinstance(value, dict):
        return {str(k): _describe(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v, seen) for v in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if id(value) in seen:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "reference": True}
    seen.add(id(value))
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, np.ndarray):
        return {
            "type": kind,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.random.Generator):
        return {"type": kind, "state": _describe(value.bit_generator.state, seen)}
    attrs = vars(value) if hasattr(value, "__dict__") else {}
    if dataclasses.is_dataclass(value):
        attrs = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    return {"type": kind, "params": _describe(attrs, seen)}


def _publish_record(record):
    # A normal backtest/search must not import the optional research stack.
    import sys

    module = sys.modules.get("tradelearn.research.run")
    run = module.current_run() if module is not None else None
    if run is not None:
        from copy import deepcopy

        run.record_step("trial", category="search.trial", params=deepcopy(record))


def execute_trial(records, params, config, execute, evaluate=None, *, publish=True):
    """One existing historical run, evaluation and retained-in-owner audit fact."""
    from copy import deepcopy

    record = {
        "format": "tradelearn.search.trial.v1",
        "index": len(records),
        "params": _describe(deepcopy(params)),
        "config": deepcopy(config),
        "status": "running",
        "evaluation": None,
    }
    records.append(record)
    token = _ACTIVE_TRIAL.set(record)
    try:
        result = execute()
        stats = (
            result[0].stats
            if isinstance(result, list) and result and hasattr(result[0], "stats")
            else result
        )
        summary = getattr(stats, "summary", None)
        if summary is not None:
            record["result_summary"] = _describe(dict(summary))
        if evaluate is not None:
            record["evaluation"] = _describe(evaluate(result))
        record["status"] = "completed"
        return result
    except BaseException as exc:
        record["status"] = "failed" if isinstance(exc, Exception) else "cancelled"
        record["error_type"] = type(exc).__name__
        # Retain the original exception and traceback; never stringify secrets.
        raise
    finally:
        _ACTIVE_TRIAL.reset(token)
        if publish:
            _publish_record(record)


def cancel_remaining(records, plans):
    from copy import deepcopy

    for item, config in plans:
        record = {
            "format": "tradelearn.search.trial.v1",
            "index": len(records),
            "params": _describe(item),
            "config": deepcopy(config),
            "status": "cancelled",
            "evaluation": None,
            "reason": "search_aborted",
        }
        records.append(record)
        _publish_record(record)


_ACTIVE_TRIAL = ContextVar("tradelearn_search_trial", default=None)


def current_trial_record():
    return _ACTIVE_TRIAL.get()
