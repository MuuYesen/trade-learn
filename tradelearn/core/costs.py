"""Commission and slippage models using signed trade directions and quote-currency costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BUY_SIDE = 1
SELL_SIDE = 2


@dataclass(frozen=True)
class FixedSlippage:
    """Shift a fill price by a fixed amount against the trading direction."""
    amount: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        """Add the fixed amount for buys and subtract it for sells."""
        adj = float(self.amount)
        return price + adj if side == BUY_SIDE else price - adj


@dataclass(frozen=True)
class PercentSlippage:
    """Apply adverse slippage as a fraction of the unadjusted fill price."""
    ratio: float = 0.0

    def apply(self, price: float, side: int, order: Any = None) -> float:
        """Shift the price by price times ratio, upward for buys and downward for sells."""
        adj = float(price) * float(self.ratio)
        return price + adj if side == BUY_SIDE else price - adj


@dataclass
class BarRangeSlippage:
    """Draw seeded adverse slippage from the current high-low range."""
    ratio: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Create the per-model random generator from the optional seed."""
        self._rng = np.random.default_rng(self.seed)

    def apply(self, price: float, side: int, order: Any = None) -> float:
        """Use a random fraction of the bar range; missing range data gives zero adjustment."""
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
    """Charge a fixed quote-currency amount per fill, independent of trade size."""
    amount: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        """Return the configured flat charge for this fill."""
        return float(self.amount)

    def as_config(self) -> float:
        """Expose the flat charge as a scalar broker configuration value."""
        return float(self.amount)


@dataclass(frozen=True)
class PercentCommission:
    """Charge a fraction of absolute traded notional."""
    ratio: float = 0.0

    def calculate(self, size: float, price: float, side: int) -> float:
        """Multiply absolute size by fill price and the commission ratio."""
        return abs(float(size)) * float(price) * float(self.ratio)

    def as_config(self) -> float:
        """Expose the notional commission ratio for broker configuration."""
        return float(self.ratio)


@dataclass(frozen=True)
class TieredCommission:
    """Select a notional commission rate from ascending threshold tiers."""
    tiers: list[tuple[float, float]]

    def calculate(self, size: float, price: float, side: int) -> float:
        """Apply the highest reached threshold rate; notional below all tiers costs zero."""
        notional = abs(float(size)) * float(price)
        ratio = 0.0
        for threshold, tier_ratio in sorted(self.tiers, key=lambda item: item[0]):
            if notional >= threshold:
                ratio = float(tier_ratio)
        return notional * ratio


@dataclass(frozen=True)
class CNAStockCommission:
    """Combine minimum brokerage, transfer fees, and sell-side stamp tax for A shares."""
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002

    def calculate(self, size: float, price: float, side: int) -> float:
        """Sum brokerage, transfer fees, and applicable stamp tax, rounded to six decimals."""
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
