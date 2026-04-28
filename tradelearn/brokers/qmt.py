"""QMT proxy broker adapter.

The adapter talks to a local quant-qmt-proxy style HTTP service.  It keeps the
same Broker protocol surface used by backtests, while making live-mode risk
checks explicit and testable before real QMT side effects are enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import httpx

from tradelearn.core.errors import TradelearnError


class QMTBrokerError(TradelearnError):
    """Raised when the QMT broker adapter rejects or cannot complete a request."""


class QMTBroker:
    """HTTP-backed Broker Protocol adapter for QMT paper/live execution."""

    _VALID_MODES = {"paper", "live"}

    def __init__(
        self,
        proxy_url: str = "http://127.0.0.1:8000",
        *,
        mode: str = "paper",
        account_id: str | None = None,
        max_order_value: float | None = None,
        require_live_confirmation: bool = True,
        timeout: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"unsupported QMT mode {mode!r}; expected {sorted(self._VALID_MODES)}")
        self.proxy_url = proxy_url.rstrip("/")
        self.mode = mode
        self.account_id = account_id
        self.max_order_value = max_order_value
        self.require_live_confirmation = require_live_confirmation
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._connected = False
        self._fill_callbacks: list[Callable[[Any], None]] = []
        self._cancel_callbacks: list[Callable[[Any], None]] = []
        self._reject_callbacks: list[Callable[[Any, str], None]] = []

    def connect(self) -> None:
        """Open connectivity to the QMT proxy."""
        response = self._client.get(f"{self.proxy_url}/health")
        self._raise_for_response(response)
        self._connected = True

    def disconnect(self) -> None:
        """Close broker connectivity."""
        self._connected = False
        if self._owns_client:
            self._client.close()

    def is_connected(self) -> bool:
        """Return whether the adapter has a live proxy connection."""
        return self._connected

    def place(self, order: Any, *, confirm_live: bool = False) -> Any:
        """Place an order through the proxy after local risk checks."""
        self._require_connected()
        payload = self._order_payload(order)
        self._validate_live_confirmation(confirm_live)
        self._validate_order_value(payload)
        response = self._client.post(f"{self.proxy_url}/orders", json=self._with_context(payload))
        self._raise_for_response(response)
        data = response.json()
        return data.get("order_id", data)

    def cancel(self, oid: Any) -> None:
        """Cancel an existing order by id."""
        self._require_connected()
        response = self._client.post(f"{self.proxy_url}/orders/{oid}/cancel", json=self._context())
        self._raise_for_response(response)

    def modify(self, oid: Any, **kwargs: Any) -> None:
        """Modify broker-supported order fields."""
        self._require_connected()
        response = self._client.patch(
            f"{self.proxy_url}/orders/{oid}",
            json=self._with_context(dict(kwargs)),
        )
        self._raise_for_response(response)

    def positions(self) -> list[Any]:
        """Return current broker positions."""
        self._require_connected()
        response = self._client.get(f"{self.proxy_url}/positions", params=self._context())
        self._raise_for_response(response)
        data = response.json()
        return data if isinstance(data, list) else data.get("positions", [])

    def account(self) -> Any:
        """Return current account state."""
        self._require_connected()
        response = self._client.get(f"{self.proxy_url}/account", params=self._context())
        self._raise_for_response(response)
        return response.json()

    def order_status(self, oid: Any) -> Any:
        """Return the current status for an order id."""
        self._require_connected()
        response = self._client.get(f"{self.proxy_url}/orders/{oid}", params=self._context())
        self._raise_for_response(response)
        return response.json()

    def on_fill(self, cb: Callable[[Any], None]) -> None:
        """Register a fill callback."""
        self._fill_callbacks.append(cb)

    def on_cancel(self, cb: Callable[[Any], None]) -> None:
        """Register a cancellation callback."""
        self._cancel_callbacks.append(cb)

    def on_reject(self, cb: Callable[[Any, str], None]) -> None:
        """Register a rejection callback."""
        self._reject_callbacks.append(cb)

    def _context(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode}
        if self.account_id is not None:
            payload["account_id"] = self.account_id
        return payload

    def _with_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, **self._context()}

    def _order_payload(self, order: Any) -> dict[str, Any]:
        if isinstance(order, Mapping):
            payload = dict(order)
        else:
            payload = {
                name: getattr(order, name)
                for name in ("symbol", "side", "type", "size", "limit", "stop", "price")
                if hasattr(order, name)
            }
        required = {"symbol", "side", "size"}
        missing = sorted(required.difference(payload))
        if missing:
            raise QMTBrokerError(f"order missing required fields: {missing}")
        payload.setdefault("type", "market")
        return payload

    def _validate_live_confirmation(self, confirm_live: bool) -> None:
        if self.mode == "live" and self.require_live_confirmation and not confirm_live:
            raise QMTBrokerError("live QMT orders require confirm_live=True")

    def _validate_order_value(self, payload: dict[str, Any]) -> None:
        if self.max_order_value is None:
            return
        price = payload.get("limit", payload.get("price"))
        if price is None:
            return
        order_value = abs(float(payload["size"]) * float(price))
        if order_value > self.max_order_value:
            raise QMTBrokerError(
                f"order value {order_value:.2f} exceeds max_order_value "
                f"{self.max_order_value:.2f}"
            )

    def _require_connected(self) -> None:
        if not self._connected:
            raise QMTBrokerError("QMT broker is not connected")

    @staticmethod
    def _raise_for_response(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise QMTBrokerError(str(exc)) from exc
