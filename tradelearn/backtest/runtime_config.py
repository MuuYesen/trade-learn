"""Internal runtime configuration shared by Lite and Engine facades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BacktestRuntimeConfig:
    """Normalized backtest runtime options.

    This object is intentionally internal to ``tradelearn.backtest``. Public
    facades keep their existing constructor APIs and translate into this shape
    before invoking the shared runtime.
    """

    cash: float
    commission: float
    commission_model: Any | None = None
    match_mode: str = "exact"
    trade_on_close: bool = False
    exactbars: bool = False
    stdstats: bool = True
    stats_mode: str = "full"

    @classmethod
    def from_owner(cls, owner: Any) -> "BacktestRuntimeConfig":
        broker = getattr(owner, "broker", None)
        return cls(
            cash=float(
                getattr(owner, "_cash", getattr(broker, "_cash", 0.0)) or 0.0
            ),
            commission=float(
                getattr(
                    owner,
                    "_commission",
                    getattr(broker, "commission_ratio", 0.0),
                )
                or 0.0
            ),
            commission_model=getattr(broker, "_commission_model", None),
            match_mode=str(getattr(owner, "match_mode", "exact")),
            trade_on_close=bool(getattr(owner, "trade_on_close", False)),
            exactbars=bool(getattr(owner, "exactbars", False)),
            stdstats=bool(getattr(owner, "stdstats", True)),
            stats_mode=str(getattr(owner, "stats_mode", "full")),
        )
