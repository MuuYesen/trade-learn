# -*- coding: UTF-8 -*-
# Adapted from reference/backtesting-main/better_ma/main.py
# Only imports are changed to prove compatibility.

import tradelearn.engine as bt


class SMA(bt.Indicator):
    lines = ("sma",)
    params = (("period", 30),)

    def __init__(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        if hasattr(line, "to_series"):
            values = line.to_series().rolling(self.p.period).mean()
            self.lines.sma = line.wrap_indicator(values, name="sma")
        else:
            self.addminperiod(self.p.period)

    def next(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        if not hasattr(line, "to_series"):
            self.lines.sma[0] = sum(line.get(size=self.p.period)) / self.p.period


class CrossOver(bt.Indicator):
    lines = ("crossover",)

    def __init__(self, *args):
        d0, d1 = args if args else (self.data0, self.data1)
        if hasattr(d0, "to_series"):
            s0 = d0.to_series()
            s1 = d1.to_series()
            diff = s0 - s1
            prev = diff.shift(1)
            values = ((diff > 0) & (prev <= 0)).astype(float) - (
                (diff < 0) & (prev >= 0)
            ).astype(float)
            self.lines.crossover = d0.wrap_indicator(values, name="crossover")
        else:
            self.data0 = d0
            self.data1 = d1
            self.addminperiod(2)

    def next(self):
        if not hasattr(self.data0, "to_series"):
            up = self.data0[0] > self.data1[0] and self.data0[-1] <= self.data1[-1]
            down = self.data0[0] < self.data1[0] and self.data0[-1] >= self.data1[-1]
            self.lines.crossover[0] = 1.0 if up else -1.0 if down else 0.0


class BetterMA(bt.Strategy):
    params = (
        ('fast_sma', 60),
        ('slow_sma', 120),
        ('closeout_limit', 0.02),
        ('target_percent', 0.3),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime

        # 计算指标 - 使用原版语法
        fast_sma = SMA(self.data.close, period=self.p.fast_sma)
        slow_sma = SMA(self.data.close, period=self.p.slow_sma)
        self.crossover = CrossOver(fast_sma, slow_sma)

        self.buy_price = None
        self.sell_price = None
        self.buy_create = None  # 金叉价格
        self.sell_create = None  # 死叉价格

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED @ {order.executed.price:.2f}")
                self.buy_price = order.executed.price
            elif order.issell():
                self.log(f"SELL EXECUTED @ {order.executed.price:.2f}")
                self.sell_price = order.executed.price

            self.bar_executed = len(self)

        self.order = None

    def next(self):
        if self.order:
            return

        if self.crossover == 1.0:  # 更新金叉价格
            self.buy_create = self.dataclose[0]
        if self.crossover == -1.0:  # 更新死叉价格
            self.sell_create = self.dataclose[0]

        # 策略逻辑
        if not self.position:
            if self.crossover == 1.0:  # 金叉
                self.order = self.order_target_percent(target=self.p.target_percent)
            elif self.crossover == -1.0:  # 死叉
                self.order = self.order_target_percent(target=-self.p.target_percent)
        else:
            if self.position.size > 0:
                if self.dataclose[0] < self.buy_create:
                    self.order = self.order_target_percent(target=0)
            else:
                if self.dataclose[0] > self.sell_create:
                    self.order = self.order_target_percent(target=0)
