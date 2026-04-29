"""Backtrader facade feature-surface strategy.

This strategy-only reference shows the currently supported
``tradelearn.compat.backtrader`` calling style without running a backtest.
"""

from __future__ import annotations

import tradelearn.compat.backtrader as bt


class FeatureSurfaceSizer(bt.Sizer):
    """Sizer example for Cerebro.addsizer / Strategy.getsizing integration."""

    params = (("stake", 10),)

    def getsizing(self, data, isbuy: bool, **kwargs) -> float:
        return float(self.p.stake)


class FeatureSurfaceAnalyzer(bt.Analyzer):
    """Analyzer example showing order, fill, trade, bar, and final stats hooks."""

    def __init__(self) -> None:
        self.orders = 0
        self.fills = 0
        self.trades = 0
        self.bars = 0
        self.summary = {}

    def on_order(self, order) -> None:
        self.orders += 1

    def on_fill(self, fill) -> None:
        self.fills += 1

    def on_trade(self, trade) -> None:
        self.trades += 1

    def on_bar(self, bar) -> None:
        self.bars += 1

    def on_end(self, stats) -> None:
        self.summary = dict(stats.summary)

    def get_analysis(self) -> dict[str, object]:
        return {
            "orders": self.orders,
            "fills": self.fills,
            "trades": self.trades,
            "bars": self.bars,
            "summary": self.summary,
        }


class FeatureSurfaceCommission(bt.CommInfoBase):
    """Commission scheme example for Cerebro.setcommission(comminfo=...)."""

    params = (
        ("commission", 0.001),
        ("mult", 1.0),
        ("margin", None),
    )


class BacktraderFeatureSurfaceStrategy(bt.Strategy):
    """Reference strategy covering the main Backtrader-compatible calls."""

    params = (
        ("fast", 10),
        ("slow", 30),
        ("risk_pct", 0.25),
        ("bracket_size", 1),
        ("use_runstop", False),
    )

    def __init__(self) -> None:
        # data / datas / data0 are bound by Cerebro before strategy init.
        self.data0_close = self.data0.close
        self.primary_close = self.data.close

        # Manual warmup remains available; indicator min_period is also inferred.
        self.addminperiod(self.p.slow)

        # Backtrader-style indicators are declared directly.  No Strategy.I(...)
        # exists in this facade.
        self.fast = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.slow = bt.ind.EMA(self.data.close, period=self.p.slow)
        self.rsi = bt.ind.RSI(self.data, period=14)
        self.atr = bt.ind.ATR(self.data, period=14)
        self.macd = bt.ind.MACD(self.data)
        self.bbands = bt.ind.BollingerBands(self.data.close, period=20)
        self.highest = bt.ind.Highest(self.data.high, period=20)
        self.lowest = bt.ind.Lowest(self.data.low, period=20)
        self.donchian = bt.ind.DonchianChannels(self.data, period=20)
        self.cross = bt.ind.CrossOver(self.fast, self.slow)

        self.order = None
        self.bracket_orders = []
        self.cash_values = []
        self.trade_pnls = []

    def start(self) -> None:
        self.started_value = self.broker.getvalue()

    def notify_order(self, order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            self.order = order
            return
        if order.status in (order.Completed, order.Canceled, order.Expired, order.Rejected):
            self.order = None

    def notify_trade(self, trade) -> None:
        if getattr(trade, "isclosed", False):
            self.trade_pnls.append(trade.pnl)

    def notify_cashvalue(self, cash: float, value: float) -> None:
        self.cash_values.append((cash, value))

    def next(self) -> None:
        if self.order:
            return

        if self.p.use_runstop and len(self) >= 100:
            self.cerebro.runstop()
            return

        # Multi-data reads are available when Cerebro has multiple feeds.
        if len(self.datas) > 1:
            secondary_close = self.datas[1].close[0]
            primary_close = self.data.close[0]
            spread = primary_close - secondary_close
        else:
            spread = 0.0

        if not self.position and self.cross[0] > 0 and spread >= 0:
            # Target helpers.
            self.order = self.order_target_percent(target=self.p.risk_pct)
            return

        if self.position and self.cross[0] < 0:
            self.order = self.order_target_size(target=0)
            return

        if not self.position and self.rsi[0] < 30:
            # Bracket helper preserves parent / OCO metadata in the facade.
            price = self.data.close[0]
            self.bracket_orders = self.buy_bracket(
                size=self.p.bracket_size,
                price=price,
                stopprice=price - 2 * self.atr[0],
                limitprice=price + 4 * self.atr[0],
            )

        if self.position and self.data.close[0] < self.bbands.bot[0]:
            self.close()

    def stop(self) -> None:
        self.stopped_value = self.broker.getvalue()
