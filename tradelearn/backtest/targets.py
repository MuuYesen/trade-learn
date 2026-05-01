"""Shared target-weight order intent helpers.

This module is intentionally runtime-neutral: it validates target portfolio
weights and turns snapshots into order intents, while facades decide how to
submit those intents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TargetWeightSnapshot:
    """Current state used to compute one target-weight delta."""

    price: float
    size: float
    mult: float = 1.0


@dataclass(frozen=True)
class TargetOrderIntent:
    """Neutral target-order intent produced from a target weight."""

    symbol: str
    target_weight: float
    current_weight: float
    delta_weight: float
    side: str
    qty: float
    data: Any | None = None


def coerce_target_weights(weights: Mapping[str, float] | pd.Series) -> dict[str, float]:
    """Return plain ``str -> float`` target weights."""

    items = weights.items() if hasattr(weights, "items") else dict(weights).items()
    return {str(symbol): float(weight) for symbol, weight in items}


def validate_target_weights(
    weights: Mapping[str, float] | pd.Series,
    known_symbols: set[str],
    *,
    unknown_label: str = "symbol(s)",
) -> tuple[dict[str, float], float]:
    """Validate target weights and return non-cash weights plus cash weight."""

    requested = coerce_target_weights(weights)
    cash_weight = float(requested.pop("cash", 0.0))
    if cash_weight < 0:
        raise ValueError("cash target weight must be non-negative")
    if any(weight < 0 for weight in requested.values()):
        raise ValueError("target weights must be non-negative")
    if float(sum(requested.values()) + cash_weight) > 1.000000000000001:
        raise ValueError("target weights plus cash must sum to <= 1")

    unknown = sorted(set(requested) - known_symbols)
    if unknown:
        raise ValueError(f"Unknown {unknown_label}: {unknown}")
    return requested, cash_weight


def build_target_weight_intents(
    weights: Mapping[str, float] | pd.Series,
    *,
    data_by_symbol: Mapping[str, Any],
    snapshots: Mapping[str, TargetWeightSnapshot],
    equity: float,
    close_missing: bool = True,
    unknown_label: str = "symbol(s)",
) -> list[TargetOrderIntent]:
    """Build sell-first order intents for target portfolio weights."""

    known_symbols = set(data_by_symbol)
    requested, _cash_weight = validate_target_weights(
        weights,
        known_symbols,
        unknown_label=unknown_label,
    )
    targets = dict(requested)
    if close_missing:
        for symbol in known_symbols - targets.keys():
            targets[symbol] = 0.0

    intents: list[TargetOrderIntent] = []
    for symbol, target_weight in targets.items():
        snapshot = snapshots[symbol]
        price = float(snapshot.price)
        mult = float(snapshot.mult)
        if price <= 0 or mult <= 0:
            continue
        current_value = float(snapshot.size) * price * mult
        current_weight = current_value / float(equity) if equity else 0.0
        target_value = float(target_weight) * float(equity)
        delta_value = target_value - current_value
        if abs(delta_value) < 1e-12:
            continue
        qty = int(abs(delta_value) / (price * mult))
        if not qty:
            continue
        side = "buy" if delta_value > 0 else "sell"
        intents.append(
            TargetOrderIntent(
                symbol=symbol,
                target_weight=float(target_weight),
                current_weight=float(current_weight),
                delta_weight=float(target_weight) - float(current_weight),
                side=side,
                qty=float(qty),
                data=data_by_symbol.get(symbol),
            )
        )

    intents.sort(key=lambda intent: (intent.side == "buy", intent.symbol))
    return intents
