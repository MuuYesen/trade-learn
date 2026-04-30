"""Backtrader-style asset-class portfolio strategy examples.

Adapted from:
reference/backtrader-portfolio-strategy/portfolio-allocation-master/src/strategies.py

The reference project contains several portfolio allocation strategies that
calculate target weights and then rebalance each data feed. These examples keep
the same Backtrader API semantics while avoiding external databases and
optimizer packages:

* static asset-class targets
* uniform allocation across available asset classes
* uniform allocation with a trend filter
* inverse-volatility allocation
"""

from __future__ import annotations

import math

import tradelearn.engine as bt

_TARGET_WEIGHT_PARAMS = (
    ("rebalance_bars", 20),
    ("lookback", 60),
    ("cash_reserve", 0.0),
    ("min_change", 0.0),
    (
        "shareclass_by_name",
        {
            "GLD": "gold",
            "DBC": "commodity",
            "SPY": "equity",
            "TLT": "bond_lt",
            "IEF": "bond_it",
        },
    ),
    ("printlog", False),
)


class _TargetWeightPortfolioBase(bt.Strategy):
    params = _TARGET_WEIGHT_PARAMS

    def __init__(self) -> None:
        self.order = None
        self.target_history = []
        self.order_history = []
        self.addminperiod(int(self.p.lookback) + 1)

    def log(self, txt, dt=None) -> None:
        if not self.p.printlog:
            return
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def notify_order(self, order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def start(self) -> None:
        self.starting_value = self._broker_value()

    def stop(self) -> None:
        self.final_value = self._broker_value()
        self.roi = (self.final_value / self.starting_value) - 1.0

    def next(self) -> None:
        if len(self) % int(self.p.rebalance_bars) != 0:
            return

        target_weights = self._target_weights()
        investable = max(0.0, 1.0 - float(self.p.cash_reserve))
        target_weights = {name: weight * investable for name, weight in target_weights.items()}
        self.target_history.append((self.datas[0].datetime.datetime(0), target_weights.copy()))

        for name, target, change in self._sell_first_targets(target_weights):
            if abs(change) <= float(self.p.min_change):
                continue
            self.log(f"Order Created for {name} to target percentage {target:.4f}")
            order = self.order_target_percent(self.getdatabyname(name), target)
            if order is not None:
                self.order = order
                self.order_history.append((name, target, change))

    def _target_weights(self) -> dict[str, float]:
        raise NotImplementedError

    def _shareclass(self, data) -> str:
        return self.p.shareclass_by_name.get(data._name, data._name)

    def _present_shareclasses(self) -> list[str]:
        seen = []
        for data in self.datas:
            shareclass = self._shareclass(data)
            if shareclass not in seen:
                seen.append(shareclass)
        return seen

    def _weights_from_assetclass_targets(self, targets: dict[str, float]) -> dict[str, float]:
        counts = {
            shareclass: sum(1 for data in self.datas if self._shareclass(data) == shareclass)
            for shareclass in targets
        }
        weights = {}
        for data in self.datas:
            shareclass = self._shareclass(data)
            count = counts.get(shareclass, 0)
            weights[data._name] = float(targets.get(shareclass, 0.0)) / count if count else 0.0
        return weights

    def _sell_first_targets(
        self, target_weights: dict[str, float]
    ) -> list[tuple[str, float, float]]:
        current = []
        portfolio_value = self._broker_value()
        for data in self.datas:
            target = float(target_weights.get(data._name, 0.0))
            position = self.getposition(data=data)
            current_value = float(position.size) * float(data.close[0])
            current_percent = current_value / portfolio_value if portfolio_value else 0.0
            current.append((data._name, target, target - current_percent))
        return sorted(current, key=lambda item: item[2])

    def _broker_value(self) -> float:
        get_value = getattr(self.broker, "get_value", None)
        if callable(get_value):
            return float(get_value())
        return float(self.broker.getvalue())


class AssetClassTargetPortfolioStrategy(_TargetWeightPortfolioBase):
    """Static all-weather style targets split across matching data feeds."""

    params = _TARGET_WEIGHT_PARAMS + (
        (
            "assetclass_targets",
            {
                "gold": 0.12,
                "commodity": 0.13,
                "equity": 0.20,
                "bond_lt": 0.15,
                "bond_it": 0.40,
            },
        ),
    )

    def _target_weights(self) -> dict[str, float]:
        return self._weights_from_assetclass_targets(dict(self.p.assetclass_targets))


class UniformAssetClassPortfolioStrategy(_TargetWeightPortfolioBase):
    """Allocate equally across the asset classes that are present."""

    def _target_weights(self) -> dict[str, float]:
        shareclasses = self._present_shareclasses()
        if not shareclasses:
            return {data._name: 0.0 for data in self.datas}
        class_weight = 1.0 / len(shareclasses)
        return self._weights_from_assetclass_targets(
            {shareclass: class_weight for shareclass in shareclasses}
        )


class TrendFilteredPortfolioStrategy(UniformAssetClassPortfolioStrategy):
    """Uniform asset-class allocation, but hold cash when price is below trend."""

    def _target_weights(self) -> dict[str, float]:
        weights = super()._target_weights()
        for data in self.datas:
            if float(data.close[0]) < self._average_close(data):
                weights[data._name] = 0.0
        return weights

    def _average_close(self, data) -> float:
        lookback = int(self.p.lookback)
        values = [float(data.close[-index]) for index in range(lookback)]
        return sum(values) / len(values)


class InverseVolatilityPortfolioStrategy(_TargetWeightPortfolioBase):
    """Allocate more weight to assets with lower recent return volatility."""

    def _target_weights(self) -> dict[str, float]:
        raw_weights = {}
        for data in self.datas:
            volatility = self._return_volatility(data)
            raw_weights[data._name] = 0.0 if volatility <= 0.0 else 1.0 / volatility
        total = sum(raw_weights.values())
        if total <= 0.0 or not math.isfinite(total):
            return {data._name: 0.0 for data in self.datas}
        return {name: weight / total for name, weight in raw_weights.items()}

    def _return_volatility(self, data) -> float:
        lookback = int(self.p.lookback)
        closes = [float(data.close[-index]) for index in range(lookback, -1, -1)]
        returns = [
            closes[index] / closes[index - 1] - 1.0
            for index in range(1, len(closes))
            if closes[index - 1] != 0.0
        ]
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(max(variance, 0.0))
