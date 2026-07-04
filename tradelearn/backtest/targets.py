"""Shared target-weight order intent helpers.

This module is intentionally runtime-neutral: it validates target portfolio
weights and turns snapshots into order intents, while facades decide how to
submit those intents.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class TargetOrderConstraints:
    """Optional trading constraints applied while building target-weight intents."""

    buy_lot_size: int = 1
    sell_lot_size: int = 1
    max_sell_qty_by_symbol: Mapping[str, float] | None = None


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
    constraints: TargetOrderConstraints | None = None,
) -> list[TargetOrderIntent]:
    """Build sell-first order intents for target portfolio weights."""

    constraints = constraints or TargetOrderConstraints()
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
        size = float(snapshot.size)
        if not math.isfinite(price) or not math.isfinite(mult) or not math.isfinite(size):
            continue
        if price <= 0 or mult <= 0:
            continue
        current_value = size * price * mult
        current_weight = current_value / float(equity) if equity else 0.0
        target_value = float(target_weight) * float(equity)
        delta_value = target_value - current_value
        if not math.isfinite(delta_value):
            continue
        if abs(delta_value) < 1e-12:
            continue
        if abs(float(target_weight)) < 1e-12 and abs(size) > 0:
            qty = abs(size)
            side = "sell" if size > 0 else "buy"
        else:
            qty = int(abs(delta_value) / (price * mult))
            if not qty:
                continue
            side = "buy" if delta_value > 0 else "sell"
        qty = _apply_order_constraints(
            symbol,
            side,
            qty,
            constraints,
            close_position=abs(float(target_weight)) < 1e-12,
        )
        if not qty:
            continue
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

    intents.sort(key=lambda intent: (intent.delta_weight, intent.symbol))
    return intents


def _apply_order_constraints(
    symbol: str,
    side: str,
    qty: float,
    constraints: TargetOrderConstraints,
    *,
    close_position: bool = False,
) -> float:
    if side == "buy":
        if close_position:
            return float(int(qty))
        return float(_round_down_lot(qty, constraints.buy_lot_size))
    max_sell_qty = None
    if constraints.max_sell_qty_by_symbol:
        max_sell_qty = constraints.max_sell_qty_by_symbol.get(str(symbol))
    if max_sell_qty is not None:
        qty = min(float(qty), max(float(max_sell_qty), 0.0))
    lot_size = int(constraints.sell_lot_size or 1)
    if lot_size > 1 and not close_position:
        qty = _round_down_lot(qty, lot_size)
    return float(qty)


def _round_down_lot(qty: float, lot_size: int) -> float:
    lot_size = int(lot_size or 1)
    if lot_size <= 1:
        return float(int(qty))
    return float(int(qty // lot_size) * lot_size)
