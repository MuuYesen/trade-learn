# -*- coding: UTF-8 -*-
# Adapted from reference/backtesting-main/better_ma/main.py
# Only imports are changed to prove compatibility.

import tradelearn.compat.backtrader as bt

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
        fast_sma = bt.ind.MovingAverageSimple(period=self.p.fast_sma)
        slow_sma = bt.ind.MovingAverageSimple(period=self.p.slow_sma)
        self.crossover = bt.ind.CrossOver(fast_sma, slow_sma)

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
