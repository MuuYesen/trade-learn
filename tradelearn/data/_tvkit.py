"""Internal tvkit qualification adapter, not a qualified Provider replacement.

tvkit 0.18.1 cannot express the existing unadjusted Bars contract. Keep its
typed split/dividend results distinct until that gap and Charting consumers
are resolved. This adapter has no streaming or execution entry point.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any


class TvkitResearchClient:
    """Finite research-only calls with explicit, instance-local credentials.

    Importing or constructing this class does not import tvkit, extract browser
    cookies, consult TVKIT_* credentials, authenticate, or start a thread.
    """

    def __init__(self, *, auth_token: str | None = None, timeout: float = 12.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if auth_token is not None and (not isinstance(auth_token, str) or not auth_token):
            raise ValueError("auth_token must be a nonempty explicit token")
        self._auth_token = auth_token or "unauthorized_user_token"
        self.timeout = float(timeout)

    def history(self, symbol: str, **kwargs: Any) -> list[Any]:
        """Synchronous scripts only; Notebook callers explicitly await history_async."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.history_async(symbol, **kwargs))
        raise RuntimeError("An event loop is running; await history_async(...) instead")

    async def history_async(
        self,
        symbol: str,
        *,
        interval: str = "1D",
        count: int = 5000,
        start: str | None = None,
        end: str | None = None,
        adjustment: str = "none",
    ) -> list[Any]:
        """Return actual typed upstream rows; never relabel adjusted prices as raw Bars."""
        try:
            from tvkit.api.chart.exceptions import EntitlementError, SeriesError
            from tvkit.api.chart.models.adjustment import Adjustment
            from tvkit.api.chart.ohlcv import OHLCV
        except ImportError as exc:
            raise ImportError(
                "tvkit candidate qualification requires tvkit==0.18.1 on Python>=3.11"
            ) from exc
        # Upstream validation happens before authentication or sockets. In the
        # pinned release Adjustment('none') fails: this is a migration blocker.
        mode = Adjustment(adjustment)
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("explicit exchange:symbol is required")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5000:
            raise ValueError("count must be an integer from 1 to 5000")
        if (start is None) != (end is None):
            raise ValueError("both start and end are required for range mode")
        query = {"adjustment": mode}
        if start is not None:
            query.update(start=start, end=end)
        else:
            query["bars_count"] = count

        async def fetch() -> list[Any]:
            # A non-None token suppresses the upstream environment/browser
            # credential fallback. Anonymous calls use its actual anonymous token.
            """Fetch once in an owned async context so cancellation closes the upstream client."""
            async with OHLCV(max_attempts=1, auth_token=self._auth_token) as client:
                return await client.get_historical_ohlcv(symbol, interval, **query)

        try:
            return await asyncio.wait_for(fetch(), timeout=self.timeout)
        except EntitlementError:
            raise PermissionError("TradingView refused the requested data entitlement") from None
        except SeriesError:
            raise ValueError("TradingView refused the symbol or interval") from None
        except TimeoutError:
            raise TimeoutError("tvkit research request timed out") from None
