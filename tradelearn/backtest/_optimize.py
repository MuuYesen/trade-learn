"""Shared parameter-grid helpers."""

from __future__ import annotations

from contextvars import ContextVar
from itertools import product
from typing import Any


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of parameter dictionaries."""

    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, item, strict=True)) for item in product(*values)]


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


_BROKER_CONFIG = (
    "_cash",
    "commission_ratio",
    "_mult",
    "match_mode",
    "_trade_on_close",
    "_commission_model",
    "_slippage_model",
    "_comminfo",
    "_comminfo_factory",
    "_fundmode",
    "_fundstartval",
)


def fresh_broker(broker):
    """Copy economic configuration only, never an engine/order/account state."""
    from copy import deepcopy

    from tradelearn.backtest.broker import RustBroker

    if type(broker) is not RustBroker:
        raise TypeError("historical search requires the historical RustBroker")
    clone = RustBroker(
        cash=broker._cash,
        commission=broker.commission_ratio,
        mult=broker._mult,
        match_mode=broker.match_mode,
    )
    for name in _BROKER_CONFIG:
        setattr(clone, name, deepcopy(getattr(broker, name)))
    clone.set_fundmode(clone._fundmode, clone._fundstartval)
    return clone


def historical_template(owner):
    """Freeze an independent in-process historical configuration for trials."""
    from copy import deepcopy

    if getattr(owner, "mode", "backtest") != "backtest" or any(
        getattr(d, "is_live", False) for d in owner.datas
    ):
        raise ValueError("search supports historical data only")
    broker = fresh_broker(owner.broker)
    memo = {id(owner.broker): broker}
    # Results retain native objects and callbacks; they are output, not input.
    for name in ("_last_results", "_last_strategy", "_last_stats", "_strategy_context"):
        value = vars(owner).get(name)
        if value is not None:
            memo[id(value)] = None
    for data in owner.datas:
        runtime = vars(data).get("runtime")
        if runtime is not None:
            memo[id(runtime)] = None
    template = deepcopy(owner, memo)
    template.broker = broker
    template.strats = [materialize_spec(spec) for spec in template.strats]
    template._runstop = False
    return template


def effective_config(owner):
    broker = owner.broker
    return {
        "data": [
            _describe(data._frame)
            | {"name": getattr(data, "_name", None), "feed": _describe(getattr(data, "p", None))}
            for data in owner.datas
        ],
        "broker": {
            "cash": broker._cash,
            "commission": broker.commission_ratio,
            **{
                name.lstrip("_"): _describe(getattr(broker, name))
                for name in _BROKER_CONFIG
                if name not in ("_cash", "commission_ratio")
            },
        },
        "strategies": [strategy_config(*spec) for spec in owner.strats],
        "execution": {
            name: _describe(getattr(owner, name, None))
            for name in (
                "match_mode",
                "trade_on_close",
                "stats_mode",
                "exactbars",
                "stdstats",
                "callback_batch",
                "p",
                "_sizer_spec",
                "analyzers",
                "observers",
                "_timers",
                "_tradingcal",
            )
        },
    }


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


def strategy_defaults(strategy):
    """Resolve declared defaults, without mutating the public strategy class."""
    from tradelearn.backtest.parameters import collect_param_defaults
    from tradelearn.lite.strategy import Strategy as LiteStrategy

    defaults = dict(collect_param_defaults(strategy)[1])
    if issubclass(strategy, LiteStrategy):
        framework_bases = set(LiteStrategy.__mro__)
        for cls in reversed(strategy.__mro__):
            if cls in framework_bases:
                continue
            defaults.update(
                {
                    name: value
                    for name, value in vars(cls).items()
                    if not name.startswith("_")
                    and not callable(value)
                    and not isinstance(value, (property, classmethod, staticmethod))
                }
            )
    return defaults


def materialize_spec(spec):
    """Own mutable default and selected values at the trial boundary only."""
    from copy import deepcopy

    strategy, args, params = spec
    return strategy, deepcopy(args), deepcopy({**strategy_defaults(strategy), **params})


def strategy_config(strategy, args, params):
    return {
        "class": _describe(strategy),
        "args": _describe(args),
        "params": _describe({**strategy_defaults(strategy), **params}),
    }
