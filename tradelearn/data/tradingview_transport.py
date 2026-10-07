"""Bounded TradingView transport owned by TradeLearn.

TradingView's chart protocol is unofficial. Anonymous access has the same
symbol/history entitlement limits as TradingView's public chart.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import threading
import time
import uuid
from urllib.parse import urlencode
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

import pandas as pd

# A timed-out OS DNS lookup cannot be cancelled by Python. Bound these daemon
# workers globally; abandoned requests retain their slot until cleanup finishes.
_HTTP_WORKERS = threading.BoundedSemaphore(4)
_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE = {}
_TOKEN_TTL = 15 * 60


class _TrackedHTTPMixin:
    def __init__(self, state):
        super().__init__()
        self._state = state

    def do_open(self, http_class, request, **kwargs):
        def connection(*args, **options):
            if self._state["cancelled"].is_set():
                raise TimeoutError("TradingView request timed out")
            conn = http_class(*args, **options)
            self._state["connection"] = conn
            return conn

        return super().do_open(connection, request, **kwargs)


class _TrackedHTTPHandler(_TrackedHTTPMixin, HTTPHandler):
    pass


class _TrackedHTTPSHandler(_TrackedHTTPMixin, HTTPSHandler):
    pass


RESOLUTIONS = {
    "in_1_minute": "1",
    "in_3_minute": "3",
    "in_5_minute": "5",
    "in_15_minute": "15",
    "in_30_minute": "30",
    "in_45_minute": "45",
    "in_1_hour": "60",
    "in_2_hour": "120",
    "in_3_hour": "180",
    "in_4_hour": "240",
    "in_daily": "1D",
    "in_weekly": "1W",
    "in_monthly": "1M",
}


def normalize_adjustment(value):
    """Validate TradingView's wire adjustment before opening any connection."""
    if not isinstance(value, str) or value.lower() not in {"none", "splits", "dividends"}:
        raise ValueError("Invalid TradingView adjustment; use none, splits or dividends")
    return value.lower()


def _frame(payload):
    return f"~m~{len(payload)}~m~{payload}"


class TradingViewTransport:
    """One connection per request, with a shared deadline and guaranteed close."""

    def __init__(self, *, username=None, password=None, timeout=12.0, connection_factory=None):
        if not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 15:
            raise ValueError("TradingView timeout must be between 0 and 15 seconds")
        self.username, self.password = username, password
        self.timeout = float(timeout)
        self._connection_factory = connection_factory

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TradingView request timed out")
        return remaining

    @staticmethod
    def _open_http(request, timeout, state):
        # Preserve urllib's proxy, TLS verification, and redirect handling.
        return build_opener(_TrackedHTTPHandler(state), _TrackedHTTPSHandler(state)).open(
            request,
            timeout=timeout,
        )

    @staticmethod
    def _interrupt_http(state):
        state["cancelled"].set()
        connection = state.get("connection")
        response = state.get("response")
        stream = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        for sock in (getattr(connection, "sock", None), stream):
            if sock is not None:
                try:
                    # Closing a buffered response can wait for its read lock.
                    # Shutdown the socket first; the worker closes its response.
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _http_json(self, url, *, deadline, data=None):
        request = Request(
            url,
            data=data,
            headers={
                "Origin": "https://www.tradingview.com",
                "Referer": "https://www.tradingview.com/",
                "User-Agent": "Mozilla/5.0",
            },
        )
        slots = _HTTP_WORKERS
        if not slots.acquire(timeout=self._remaining(deadline)):
            raise TimeoutError("TradingView request timed out")
        state = {"cancelled": threading.Event()}
        finished = threading.Event()

        def run():
            response = None
            try:
                response = self._open_http(request, self._remaining(deadline), state)
                state["response"] = response
                if state["cancelled"].is_set():
                    raise TimeoutError("TradingView request timed out")
                state["result"] = json.loads(response.read(2_000_000))
            except Exception as exc:
                state["error"] = exc
            finally:
                for resource in (response, state.get("connection")):
                    if resource is not None:
                        try:
                            resource.close()
                        except Exception:
                            pass
                slots.release()
                finished.set()

        worker = threading.Thread(target=run, name="tradelearn-tv-http", daemon=True)
        try:
            worker.start()
        except Exception:
            slots.release()
            raise
        try:
            if not finished.wait(self._remaining(deadline)):
                raise TimeoutError("TradingView request timed out")
            self._remaining(deadline)
            if "error" in state:
                raise state["error"]
            return state["result"]
        finally:
            if not finished.is_set():
                self._interrupt_http(state)

    def _token(self, deadline):
        if not self.username or not self.password:
            return "unauthorized_user_token"
        # Retain no username/password in the process-wide cache keys.
        key = hashlib.sha256(json.dumps([self.username, self.password]).encode()).digest()
        if not _TOKEN_LOCK.acquire(timeout=self._remaining(deadline)):
            raise TimeoutError("TradingView request timed out")
        try:
            now = time.monotonic()
            for expired in [k for k, (_, expiry) in _TOKEN_CACHE.items() if expiry <= now]:
                del _TOKEN_CACHE[expired]
            if key in _TOKEN_CACHE:
                return _TOKEN_CACHE[key][0]
            result = self._http_json(
                "https://www.tradingview.com/accounts/signin/",
                deadline=deadline,
                data=urlencode(
                    {"username": self.username, "password": self.password, "remember": "on"}
                ).encode(),
            )
            token = result.get("user", {}).get("auth_token")
            if not token:
                raise ConnectionError("TradingView authentication failed")
            self._remaining(deadline)
            if len(_TOKEN_CACHE) >= 64:
                del _TOKEN_CACHE[next(iter(_TOKEN_CACHE))]
            _TOKEN_CACHE[key] = (token, time.monotonic() + _TOKEN_TTL)
            return token
        finally:
            _TOKEN_LOCK.release()

    def _request(
        self, symbol, *, resolution=None, count=300, to=None, forward=False, adjustment="splits"
    ):
        adjustment = normalize_adjustment(adjustment)
        if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 200:
            raise ValueError("TradingView symbol is required")
        deadline = time.monotonic() + self.timeout
        ws = None
        try:
            from websocket import WebSocketTimeoutException, create_connection
        except ImportError:
            raise ConnectionError("TradingView websocket-client dependency is required") from None
        try:
            token = self._token(deadline)
            ws = (self._connection_factory or create_connection)(
                "wss://data.tradingview.com/socket.io/websocket",
                origin="https://www.tradingview.com",
                timeout=self._remaining(deadline),
            )
            session = "cs_" + uuid.uuid4().hex[:12]

            def send(method, params):
                ws.settimeout(self._remaining(deadline))
                ws.send(
                    _frame(
                        json.dumps(
                            {"m": method, "p": params}, separators=(",", ":"), ensure_ascii=True
                        )
                    )
                )

            send("set_auth_token", [token])
            send("chart_create_session", [session, ""])
            send(
                "resolve_symbol",
                [
                    session,
                    "symbol_1",
                    "="
                    + json.dumps(
                        {
                            "symbol": symbol,
                            "adjustment": adjustment,
                            "session": "regular",
                        },
                        separators=(",", ":"),
                    ),
                ],
            )
            if resolution is not None:
                # Negative bar_count includes an exact anchor; advance one second
                # so replay always starts with the subsequent candle.
                anchor = to + 1 if forward else to
                window = (
                    count if to is None else ["bar_count", anchor, -count if forward else count]
                )
                send("create_series", [session, "s1", "s1", "symbol_1", resolution, window])
                send("switch_timezone", [session, "Etc/UTC"])
            buffer, bars = "", {}
            while True:
                ws.settimeout(self._remaining(deadline))
                chunk = ws.recv()
                if not chunk:
                    raise ConnectionError("TradingView connection closed before completion")
                buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
                if len(buffer) > 4_000_000:
                    raise ConnectionError("TradingView response exceeded size limit")
                while buffer:
                    match = re.match(r"~m~(\d+)~m~", buffer)
                    if match is None:
                        if len(buffer) < 24 and (
                            "~m~".startswith(buffer) or buffer.startswith("~m~")
                        ):
                            break
                        raise ConnectionError("TradingView malformed response")
                    length = int(match[1])
                    end = match.end() + length
                    if len(buffer) < end:
                        break
                    payload, buffer = buffer[match.end() : end], buffer[end:]
                    if payload.startswith("~h~"):
                        ws.settimeout(self._remaining(deadline))
                        ws.send(_frame(payload))
                        continue
                    packet = json.loads(payload)
                    if not isinstance(packet, dict):
                        continue  # Initial socket session identifier.
                    method, params = packet.get("m"), packet.get("p", [])
                    if method in {
                        "symbol_error",
                        "series_error",
                        "critical_error",
                        "protocol_error",
                        "error",
                    }:
                        raise ConnectionError("TradingView request failed")
                    if method == "symbol_resolved" and resolution is None:
                        return {"series_id": params[1], **params[2]}
                    if method in {"timescale_update", "du"}:
                        for item in params[1].get("s1", {}).get("s", []):
                            values = item.get("v", [])
                            if len(values) < 5 or any(
                                v is None or not math.isfinite(float(v)) for v in values[:5]
                            ):
                                continue
                            timestamp = int(values[0])
                            volume = values[5] if len(values) > 5 else None
                            volume = float(volume) if volume is not None else None
                            if volume is not None and (not math.isfinite(volume) or volume < 0):
                                volume = None
                            bars[timestamp] = dict(
                                zip(
                                    ("time", "open", "high", "low", "close", "volume"),
                                    (timestamp, *map(float, values[1:5]), volume),
                                    strict=True,
                                )
                            )
                    if method == "series_completed":
                        return [bars[key] for key in sorted(bars)]
        except (TimeoutError, WebSocketTimeoutException):
            raise TimeoutError("TradingView request timed out") from None
        except Exception:
            raise ConnectionError("TradingView request failed") from None
        finally:
            if ws is not None:
                try:
                    ws.close(timeout=0)
                except Exception:
                    pass

    def history_window(
        self, symbol, resolution, count=300, to=None, forward=False, *, adjustment="splits"
    ):
        """Return ascending epoch-second OHLCV rows; forward excludes its anchor."""
        if isinstance(count, bool) or int(count) != count or not 1 <= count <= 5000:
            raise ValueError("TradingView count must be an integer from 1 to 5000")
        if to is not None:
            if isinstance(to, bool) or not math.isfinite(float(to)) or float(to) < 0:
                raise ValueError("TradingView cutoff must be a nonnegative epoch timestamp")
            to = int(to)
        if forward and to is None:
            raise ValueError("TradingView forward history requires a cutoff")
        resolution = str(resolution)
        if not re.fullmatch(r"(?:[1-9]\d{0,3}|[1-9]\d{0,2}[DWM]|[DWM])", resolution):
            raise ValueError("Unsupported TradingView resolution")
        rows = self._request(
            symbol,
            resolution=resolution,
            count=int(count),
            to=to,
            forward=forward,
            adjustment=adjustment,
        )
        if to is not None:
            rows = (
                [row for row in rows if row["time"] > to]
                if forward
                else [row for row in rows if row["time"] <= to]
            )
        return rows[:count] if forward else rows[-count:]

    def resolve_symbol(self, symbol, *, adjustment="splits"):
        """Return TradingView's original chart metadata."""
        return self._request(symbol, adjustment=adjustment)

    def search_symbols(self, query):
        """Return the serializable fields of TradingView.searchMarket results."""
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("TradingView search query is required")
        try:
            rows = self._http_json(
                "https://symbol-search.tradingview.com/symbol_search?"
                + urlencode({"text": query, "type": ""}),
                deadline=time.monotonic() + self.timeout,
            )
            return [
                {
                    "id": row["exchange"].split(" ")[0] + ":" + row["symbol"],
                    "exchange": row["exchange"].split(" ")[0],
                    "fullExchange": row["exchange"],
                    "symbol": row["symbol"],
                    "description": row.get("description", ""),
                    "type": row.get("type", ""),
                }
                for row in rows
            ]
        except TimeoutError:
            raise TimeoutError("TradingView search timed out") from None
        except Exception:
            raise ConnectionError("TradingView search failed") from None

    def get_hist(
        self, symbol, exchange, interval="1D", n_bars=5000, *, to=None, adjustment="splits"
    ):
        """tvDatafeed-compatible DataFrame, including a real server-side cutoff."""
        resolution = RESOLUTIONS.get(
            getattr(interval, "name", interval), getattr(interval, "value", interval)
        )
        rows = self.history_window(
            f"{exchange}:{symbol}", resolution, count=n_bars, to=to, adjustment=adjustment
        )
        frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        frame["datetime"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
        return frame.set_index("datetime")
