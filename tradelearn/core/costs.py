from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BUY_SIDE = 1
SELL_SIDE = 2


@dataclass(frozen=True)
class FixedSlippage:
    amount: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        adj = float(self.amount)
        return price + adj if side == BUY_SIDE else price - adj


@dataclass(frozen=True)
class PercentSlippage:
    ratio: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        adj = float(price) * float(self.ratio)
        return price + adj if side == BUY_SIDE else price - adj


@dataclass
class BarRangeSlippage:
    ratio: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def apply(self, price: float, side: int, order: Any = None) -> float:
        data = getattr(order, "data", None)
        high = getattr(data, "high", None)
        low = getattr(data, "low", None)
        try:
            bar_range = float(high[0]) - float(low[0])
        except Exception:
            bar_range = 0.0
        adj = float(self._rng.random()) * bar_range * float(self.ratio)
        slipped = price + adj if side == BUY_SIDE else price - adj
        return round(slipped, 6)


@dataclass(frozen=True)
class FixedCommission:
    amount: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        return float(self.amount)

    def as_config(self) -> float:
        return float(self.amount)


@dataclass(frozen=True)
class PercentCommission:
    ratio: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        return abs(float(size)) * float(price) * float(self.ratio)

    def as_config(self) -> float:
        return float(self.ratio)


@dataclass(frozen=True)
class TieredCommission:
    tiers: list[tuple[float, float]]

    def calculate(self, size: float, price: float, side: int) -> float:
        notional = abs(float(size)) * float(price)
        ratio = 0.0
        for threshold, tier_ratio in sorted(self.tiers, key=lambda item: item[0]):
            if notional >= threshold:
                ratio = float(tier_ratio)
        return notional * ratio


@dataclass(frozen=True)
class CNAStockCommission:
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002

    def calculate(self, size: float, price: float, side: int) -> float:
        notional = abs(float(size)) * float(price)
        commission = max(notional * self.commission_rate, self.min_commission)
        transfer_fee = notional * self.transfer_fee_rate
        stamp_tax = notional * self.stamp_tax_rate if side == SELL_SIDE else 0.0
        return round(commission + transfer_fee + stamp_tax, 6)


SlippageModel = FixedSlippage | PercentSlippage | BarRangeSlippage
CommissionModel = (
    FixedCommission | PercentCommission | TieredCommission | CNAStockCommission
)


__all__ = [
    "BUY_SIDE",
    "SELL_SIDE",
    "BarRangeSlippage",
    "CNAStockCommission",
    "CommissionModel",
    "FixedCommission",
    "FixedSlippage",
    "PercentCommission",
    "PercentSlippage",
    "SlippageModel",
    "TieredCommission",
]
