"""Backtrader-style target-percent portfolio rotation strategy.

Adapted from:
reference/backtrader-portfolio-strategy/PortfolioBacktesting-main/MaxSharpeStrategy.py

The reference strategy periodically computes target weights, sorts orders so
reductions are submitted before increases, and uses ``order_target_percent`` for
each named data feed. This version keeps that portfolio API shape while using a
deterministic momentum allocator so it can run without external optimizers or
CSV side files.
"""

from __future__ import annotations

import tradelearn.engine as bt


class TargetPercentPortfolioStrategy(bt.Strategy):
    """Rotate a multi-asset portfolio with Backtrader target-percent orders."""

    params = (
        ("rebalance_bars", 15),
        ("lookback", 20),
        ("top_n", 3),
        ("cash_reserve", 0.1),
        ("min_change", 0.0),
        ("printlog", False),
    )

    def __init__(self) -> None:
        self.order = None
        self.target_history = []
        self.order_history = []
        self.addminperiod(self.p.lookback + 1)

    def log(self, txt, dt=None) -> None:
        if not self.p.printlog:
            return
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def notify_order(self, order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{order.data._name} {side} EXECUTED, "
                f"Price: {order.executed.price:.2f}, "
                f"Size: {order.executed.size:.2f}, "
                f"Comm {order.executed.comm:.2f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade) -> None:
        if getattr(trade, "isclosed", False):
            self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def start(self) -> None:
        self.starting_value = self._broker_value()

    def stop(self) -> None:
        self.final_value = self._broker_value()
        self.roi = (self.final_value / self.starting_value) - 1.0

    def next(self) -> None:
        if len(self) % int(self.p.rebalance_bars) != 0:
            return

        target_weights = self._target_weights()
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
        scores = []
        for data in self.datas:
            past = float(data.close[-int(self.p.lookback)])
            current = float(data.close[0])
            momentum = current / past - 1.0 if past else 0.0
            scores.append((data._name, momentum))

        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        selected = [name for name, score in ranked[: int(self.p.top_n)] if score > 0.0]
        target_weights = {data._name: 0.0 for data in self.datas}
        if not selected:
            return target_weights

        investable_weight = max(0.0, 1.0 - float(self.p.cash_reserve))
        per_asset = investable_weight / len(selected)
        for name in selected:
            target_weights[name] = per_asset
        return target_weights

    def _sell_first_targets(
        self, target_weights: dict[str, float]
    ) -> list[tuple[str, float, float]]:
        current = []
        portfolio_value = self._broker_value()
        for name, target in target_weights.items():
            data = self.getdatabyname(name)
            position = self.getposition(data=data)
            current_value = float(position.size) * float(data.close[0])
            current_percent = current_value / portfolio_value if portfolio_value else 0.0
            current.append((name, float(target), float(target) - current_percent))
        return sorted(current, key=lambda item: item[2])

    def _broker_value(self) -> float:
        get_value = getattr(self.broker, "get_value", None)
        if callable(get_value):
            return float(get_value())
        return float(self.broker.getvalue())
