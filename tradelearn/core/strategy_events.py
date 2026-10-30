"""Read-only strategy-domain observations, independent of SDKs and wire schemas.

No identity, event ID, revision, durable log, transport or financial computation
is created here. External observers bind their own validated run/environment.
"""

from __future__ import annotations

import inspect
import warnings
from collections import deque
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import wraps
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

OBSERVER_API_VERSION = 1

_ACTIVE_OBSERVER = ContextVar("tradelearn_strategy_observer", default=False)
_CALLBACK_NAMES = {
    "_setup": "lifecycle.setup",
    "init": "lifecycle.init",
    "start": "lifecycle.start",
    "stop": "lifecycle.stop",
    "next": "business.next",
    "prenext": "business.prenext",
    "notify_order": "business.notify_order",
    "notify_trade": "business.notify_trade",
    "notify_cashvalue": "business.notify_cashvalue",
    "_pre_next": "business.pre_next",
}
_OBJECT_FIELDS = frozenset(
    (
        "ref",
        "status",
        "size",
        "price",
        "exectype",
        "data",
        "executed",
        "value",
        "comm",
        "commission",
        "pnl",
        "pnlcomm",
        "isopen",
        "isclosed",
        "dt",
        "dtopen",
        "dtclose",
        "name",
        "_name",
        "ts",
        "ts_event",
        "ts_init",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "order_id",
        "client_oid",
        "side",
        "info",
        "ordtype",
        "time_in_force",
        "pricelimit",
        "created_ts",
        "raw",
        "precision",
        "code",
        "currency",
        "amount",
        "ts_event_ns",
        "sequence",
        "event",
    )
)


def assert_observer_allows_mutation():
    """Reject framework mutations while an observer callback is active."""
    if _ACTIVE_OBSERVER.get():
        raise RuntimeError("strategy observer cannot invoke state-changing framework APIs")


def observer_mutation_guard(callback):
    """Wrap a callback with the observer mutation guard once."""
    if getattr(callback, "__tl_observer_guard__", False):
        return callback

    @wraps(callback)
    def guarded(*args, **kwargs):
        """Check observer context before calling the wrapped callback."""
        assert_observer_allows_mutation()
        return callback(*args, **kwargs)

    guarded.__tl_observer_guard__ = True
    return guarded


def _freeze(value, depth=0):
    """Convert supported domain values into bounded immutable snapshots."""
    if value is None or isinstance(value, (str, bytes, bool, date, datetime, timedelta)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if depth >= 8:
        raise ValueError("strategy snapshot nesting budget exceeded")
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v, depth + 1) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v, depth + 1) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v, depth + 1) for v in value)
    # Explicit extension protocol; it must return domain values, not wire envelopes.
    protocol = getattr(value, "__strategy_snapshot__", None)
    if callable(protocol):
        snapshot = protocol()
        if not isinstance(snapshot, Mapping):
            raise TypeError("strategy snapshot protocol must return a mapping")
        return _freeze(snapshot, depth + 1)
    fields = vars(value) if hasattr(value, "__dict__") else {}
    result = {"__type__": type(value).__module__ + "." + type(value).__qualname__}
    if "_buffer" in fields or "_frame" in fields:
        result["_name"] = _freeze(fields.get("_name"), depth + 1)
        return MappingProxyType(result)  # Never materialize a full data/indicator window.
    for key in _OBJECT_FIELDS:
        try:
            item = fields[key] if key in fields else getattr(value, key)
        except AttributeError:
            continue
        if callable(item):
            continue
        result[key] = _freeze(item, depth + 1)
    if len(result) == 1:
        raise TypeError("unsupported strategy snapshot object: " + result["__type__"])
    return MappingProxyType(result)


@dataclass(frozen=True)
class StrategyEvent:
    """Describe one read-only lifecycle, business, or intent observation."""
    name: str
    phase: str
    details: Mapping[str, Any]
    context: Mapping[str, Any]
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ObserverSpec:
    """Configure an observer, event filters, context, and error handler."""
    observer: Callable[[StrategyEvent], None]
    events: frozenset[str] | None = None
    phases: tuple[str, ...] = ("after", "error")
    context: Mapping[str, Any] | None = None
    error_handler: Callable[[Any], None] | None = None


@dataclass(frozen=True)
class ObservationFailure:
    """Record an observer or snapshot failure for diagnostics."""
    event: StrategyEvent
    observer: str
    error_type: str
    error_message: str
    source: str = "observer"


@dataclass(frozen=True)
class ObservationFailures:
    """Summarize retained and omitted observation failures."""
    total: int
    omitted: int
    retained: tuple[ObservationFailure, ...]


@dataclass(eq=False)
class _Subscription:
    """Hold a registered observer specification and warning state."""
    spec: ObserverSpec
    warned: bool = False


class _State:
    """Track observer subscriptions and bounded failure diagnostics."""
    def __init__(self):
        self.subscriptions = []
        self.failures = deque(maxlen=64)
        self.failure_count = 0
        self.omitted_count = 0


def _state(strategy, create=False):
    """Retrieve or initialize observation state for a strategy."""
    state = getattr(strategy, "_strategy_observation_state", None)
    if state is None and create:
        state = _State()
        strategy._strategy_observation_state = state
    return state


def add_observer(
    strategy, observer, *, events=None, phases=("after", "error"), context=None, error_handler=None
):
    """Validate and register a synchronous observer on a strategy."""
    assert_observer_allows_mutation()
    if inspect.iscoroutinefunction(observer) or inspect.iscoroutinefunction(
        inspect.getattr_static(type(observer), "__call__", None)
    ):
        raise TypeError("strategy observers must be synchronous")
    if not callable(observer):
        raise TypeError("observer must be callable")
    if error_handler is not None and not callable(error_handler):
        raise TypeError("error_handler must be callable")
    phases = tuple(phases)
    if not phases or any(p not in ("before", "after", "error") for p in phases):
        raise ValueError("observer phases must be before/after/error")
    if events is not None:
        if isinstance(events, str):
            raise TypeError("events must be a collection of domain names")
        events = frozenset(events)
        if any(not isinstance(e, str) for e in events):
            raise TypeError("event names must be strings")
    spec = ObserverSpec(observer, events, phases, _freeze(context or {}), error_handler)
    subscription = _Subscription(spec)
    _state(strategy, True).subscriptions.append(subscription)
    return subscription


def remove_observer(strategy, subscription):
    """Remove a subscription by identity and report whether it existed."""
    assert_observer_allows_mutation()
    state = _state(strategy)
    if state is None:
        return False
    for index, current in enumerate(state.subscriptions):
        if current is subscription:
            state.subscriptions.pop(index)
            return True
    return False


def has_observers(strategy, name=None):
    """Report whether an observer is interested in a domain event."""
    state = _state(strategy)
    return bool(
        state
        and any(
            s.spec.events is None or name is None or name in s.spec.events
            for s in state.subscriptions
        )
    )


def invoke_component(strategy, callback, *args, **kwargs):
    """Invoke a component through its runtime when one is bound."""
    runtime = getattr(strategy, "_runtime", None)
    if runtime is not None:
        return runtime.invoke_component(callback, *args, **kwargs)
    assert_observer_allows_mutation()
    return callback(*args, **kwargs)


def observation_failures(strategy):
    """Return bounded diagnostic failures and cumulative counts."""
    state = _state(strategy)
    if state is None:
        return ObservationFailures(0, 0, ())
    return ObservationFailures(state.failure_count, state.omitted_count, tuple(state.failures))


def _error_message(error):
    """Render an exception without letting rendering mask the original failure."""
    try:
        return str(error)
    except BaseException:
        return type(error).__name__


def _record_failure(state, subscription, event, error, source="observer"):
    """Retain one observer failure and warn at most once per subscription."""
    try:
        name = getattr(
            subscription.spec.observer,
            "__qualname__",
            type(subscription.spec.observer).__qualname__,
        )
    except BaseException:
        name = "observer"
    failure = ObservationFailure(event, name, type(error).__name__, _error_message(error), source)
    if len(state.failures) == state.failures.maxlen:
        state.omitted_count += 1
    state.failures.append(failure)
    state.failure_count += 1
    if not subscription.warned:
        subscription.warned = True
        try:
            warnings.warn(
                "TradeLearn strategy observer failed; inspect observation_failures(): "
                + _error_message(error),
                RuntimeWarning,
                stacklevel=3,
            )
        except BaseException:
            pass  # Warning hooks/filters must never replace a business exception.
    return failure


def emit_domain(strategy, name, phase, details=None, *, error=None):
    """Dispatch an immutable domain snapshot to matching observers."""
    assert_observer_allows_mutation()
    if not isinstance(name, str) or not name.startswith(("lifecycle.", "business.", "intent.")):
        raise ValueError("TL observations are lifecycle/business/intent, never execution authority")
    if phase not in ("before", "after", "error"):
        raise ValueError("invalid domain observation phase")
    state = _state(strategy)
    if state is None or not state.subscriptions:
        return
    interested = tuple(
        s
        for s in state.subscriptions
        if phase in s.spec.phases and (s.spec.events is None or name in s.spec.events)
    )
    if not interested:
        return
    token = _ACTIVE_OBSERVER.set(True)
    try:
        frozen = _freeze(details or {})
    except BaseException as snapshot_error:
        for subscription in interested:
            event = StrategyEvent(name, phase, MappingProxyType({}), subscription.spec.context)
            _record_failure(state, subscription, event, snapshot_error, "snapshot")
        return
    finally:
        _ACTIVE_OBSERVER.reset(token)
    for subscription in interested:
        event = StrategyEvent(
            name,
            phase,
            frozen,
            subscription.spec.context,
            type(error).__name__ if error is not None else None,
            _error_message(error) if error is not None else None,
        )
        token = _ACTIVE_OBSERVER.set(True)
        try:
            try:
                result = subscription.spec.observer(event)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError(
                        "strategy observers must be synchronous; awaitable result rejected"
                    )
            except BaseException as failure:
                record = _record_failure(state, subscription, event, failure)
                if subscription.spec.error_handler is not None:
                    try:
                        result = subscription.spec.error_handler(record)
                        if inspect.isawaitable(result):
                            if inspect.iscoroutine(result):
                                result.close()
                            raise TypeError("observer error handlers must be synchronous")
                    except BaseException as handler_failure:
                        _record_failure(
                            state, subscription, event, handler_failure, "error_handler"
                        )
        finally:
            _ACTIVE_OBSERVER.reset(token)


def invoke_callback(strategy, callback_name, *args, **kwargs):
    """Invoke a callback while reporting its observation phases."""
    assert_observer_allows_mutation()
    callback = getattr(strategy, callback_name)
    state = _state(strategy)
    if state is None or not state.subscriptions:
        return callback(*args, **kwargs)
    name = _CALLBACK_NAMES.get(callback_name, "business." + callback_name)
    details = {"args": args, "kwargs": kwargs}
    runtime_context = getattr(strategy, "runtime_context", None)
    if runtime_context is not None:
        details["runtime_context"] = runtime_context
    emit_domain(strategy, name, "before", details)
    try:
        result = callback(*args, **kwargs)
    except BaseException as error:
        emit_domain(strategy, name, "error", details, error=error)
        raise
    emit_domain(strategy, name, "after", details)
    return result


def dispatch_callback(strategy, callback_name, *args, **kwargs):
    """Dispatch a named strategy callback through the observation boundary."""
    runtime = getattr(strategy, "_runtime", None)
    if runtime is not None:
        return runtime.invoke_business(callback_name, *args, **kwargs)
    return invoke_callback(strategy, callback_name, *args, **kwargs)


def assemble_observers(strategy):
    """Call inherited neutral factories once, without requiring user super()."""
    assert_observer_allows_mutation()
    if getattr(strategy, "_strategy_observers_assembled", False):
        return
    strategy._strategy_observers_assembled = True
    seen = set()
    for cls in reversed(type(strategy).__mro__):
        for factory in vars(cls).get("strategy_observer_factories", ()):
            if id(factory) in seen:
                continue
            seen.add(id(factory))
            # Assembly is an observer extension, not another strategy lifecycle.
            token = _ACTIVE_OBSERVER.set(True)
            try:
                result = factory(strategy)
            finally:
                _ACTIVE_OBSERVER.reset(token)
            spec = result if isinstance(result, ObserverSpec) else ObserverSpec(result)
            add_observer(
                strategy,
                spec.observer,
                events=spec.events,
                phases=spec.phases,
                context=spec.context,
                error_handler=spec.error_handler,
            )


__all__ = [
    "OBSERVER_API_VERSION",
    "StrategyEvent",
    "ObserverSpec",
    "ObservationFailure",
    "ObservationFailures",
    "add_observer",
    "remove_observer",
    "observation_failures",
]
