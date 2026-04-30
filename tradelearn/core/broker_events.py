"""Generic broker proxy event polling and dispatch helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

BrokerEventKind = Literal["fill", "cancel", "reject", "status", "partial"]


@dataclass(frozen=True)
class BrokerEvent:
    """Event emitted by a broker proxy poller."""

    kind: BrokerEventKind
    order_id: Any
    payload: Any = None
    reason: str | None = None
    status: str | None = None
    replay: bool = False
    requires_confirmation: bool = False
    max_notional: float | None = None
    risk_tags: tuple[str, ...] = ()

    @classmethod
    def coerce(cls, raw_event: BrokerEvent | dict[str, Any]) -> BrokerEvent:
        """Normalize a raw broker event dict into a broker event object."""
        if isinstance(raw_event, BrokerEvent):
            return raw_event
        return cls(
            kind=raw_event["kind"],
            order_id=raw_event.get("order_id"),
            payload=raw_event.get("payload"),
            reason=raw_event.get("reason"),
            status=raw_event.get("status"),
            replay=bool(raw_event.get("replay", False)),
            requires_confirmation=bool(raw_event.get("requires_confirmation", False)),
            max_notional=raw_event.get("max_notional"),
            risk_tags=tuple(raw_event.get("risk_tags", ())),
        )


@dataclass
class BrokerEventPump:
    """Poll broker proxy events and dispatch them to registered callbacks."""

    poller: Callable[[], Iterable[BrokerEvent | dict[str, Any]]]
    _fill_callbacks: list[Callable[[Any], None]] = field(default_factory=list)
    _cancel_callbacks: list[Callable[[Any], None]] = field(default_factory=list)
    _reject_callbacks: list[Callable[[Any, str], None]] = field(default_factory=list)
    _status_callbacks: list[Callable[[Any, str, bool], None]] = field(default_factory=list)
    _partial_callbacks: list[Callable[[Any], None]] = field(default_factory=list)
    _event_callbacks: list[Callable[[BrokerEvent], None]] = field(default_factory=list)

    def on_fill(self, cb: Callable[[Any], None]) -> None:
        self._fill_callbacks.append(cb)

    def on_cancel(self, cb: Callable[[Any], None]) -> None:
        self._cancel_callbacks.append(cb)

    def on_reject(self, cb: Callable[[Any, str], None]) -> None:
        self._reject_callbacks.append(cb)

    def on_status(self, cb: Callable[[Any, str, bool], None]) -> None:
        self._status_callbacks.append(cb)

    def on_partial(self, cb: Callable[[Any], None]) -> None:
        self._partial_callbacks.append(cb)

    def on_event(self, cb: Callable[[BrokerEvent], None]) -> None:
        """Register a callback that receives every normalized broker event."""
        self._event_callbacks.append(cb)

    def poll_once(self) -> int:
        """Poll once and return the number of dispatched events."""
        return len(self.poll_events())

    def poll_events(self) -> tuple[BrokerEvent, ...]:
        """Poll once, dispatch events, and return the normalized event snapshot."""
        events = tuple(self._coerce_event(raw_event) for raw_event in self.poller())
        for event in events:
            self.dispatch(event)
        return events

    def dispatch(self, event: BrokerEvent) -> None:
        for cb in self._event_callbacks:
            cb(event)
        if event.kind == "fill":
            payload = event.payload if event.payload is not None else event
            for cb in self._fill_callbacks:
                cb(payload)
        elif event.kind == "cancel":
            for cb in self._cancel_callbacks:
                cb(event.order_id)
        elif event.kind == "reject":
            reason = event.reason or ""
            for cb in self._reject_callbacks:
                cb(event.order_id, reason)
        elif event.kind == "status":
            status = event.status or ""
            for cb in self._status_callbacks:
                cb(event.order_id, status, event.replay)
        elif event.kind == "partial":
            payload = event.payload if event.payload is not None else event
            for cb in self._partial_callbacks:
                cb(payload)
        else:
            raise ValueError(f"unsupported broker event kind: {event.kind!r}")

    @staticmethod
    def _coerce_event(raw_event: BrokerEvent | dict[str, Any]) -> BrokerEvent:
        return BrokerEvent.coerce(raw_event)
