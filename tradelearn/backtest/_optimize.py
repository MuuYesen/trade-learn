"""Shared parameter-grid helpers."""

from __future__ import annotations

from itertools import product
from typing import Any

from tradelearn.backtest.runtime_config import BacktestRuntimeConfig
from tradelearn.core.trial_records import (
    _describe as _describe,
)
from tradelearn.core.trial_records import (
    _publish_record as _publish_record,
)
from tradelearn.core.trial_records import (
    cancel_remaining as cancel_remaining,
)
from tradelearn.core.trial_records import (
    current_trial_record as current_trial_record,
)
from tradelearn.core.trial_records import (
    execute_trial as execute_trial,
)


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of parameter dictionaries."""

    keys = list(param_grid)
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, item, strict=True)) for item in product(*values)]


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
    template.strats = [
        materialize_spec(spec, memo=feed_memo(owner, template)) for spec in template.strats
    ]
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
        }
        | {"periods": _describe(BacktestRuntimeConfig.from_owner(owner).periods)},
    }


def strategy_defaults(strategy):
    """Resolve declared defaults, without mutating the public strategy class."""
    from tradelearn.backtest.parameters import collect_param_defaults

    defaults = dict(collect_param_defaults(strategy)[1])
    hook = getattr(strategy, "_trial_param_defaults", None)
    if callable(hook):
        defaults.update(hook())
    return defaults


def feed_memo(source, target):
    """Keep parameter references to owned feeds on the same isolated graph."""
    memo = {id(old): new for old, new in zip(source.datas, target.datas)}
    memo.update({id(data): data for data in target.datas})
    return memo


def materialize_spec(spec, *, memo=None):
    """Own mutable default and selected values at the trial boundary only."""
    from copy import deepcopy

    strategy, args, params = spec
    memo = {} if memo is None else dict(memo)
    return strategy, deepcopy(args, memo), deepcopy({**strategy_defaults(strategy), **params}, memo)


def strategy_config(strategy, args, params):
    return {
        "class": _describe(strategy),
        "args": _describe(args),
        "params": _describe({**strategy_defaults(strategy), **params}),
    }
