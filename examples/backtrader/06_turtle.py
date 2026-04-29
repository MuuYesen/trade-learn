# -*- coding: UTF-8 -*-
# Adapted from reference/backtesting-main/turtle_trading/main.py
# Only imports are changed to prove compatibility.

import numpy as np
import pandas as pd

import tradelearn.engine as bt


class Highest(bt.Indicator):
    lines = ("highest",)
    params = (("period", 30),)

    def __init__(self):
        if hasattr(self.data, "to_series"):
            values = self.data.to_series().rolling(self.p.period).max()
            self.lines.highest = self.data.wrap_indicator(values, name="highest")
        else:
            self.addminperiod(self.p.period)

    def next(self):
        if not hasattr(self.data, "to_series"):
            self.lines.highest[0] = max(self.data.get(size=self.p.period))


class Lowest(bt.Indicator):
    lines = ("lowest",)
    params = (("period", 30),)

    def __init__(self):
        if hasattr(self.data, "to_series"):
            values = self.data.to_series().rolling(self.p.period).min()
            self.lines.lowest = self.data.wrap_indicator(values, name="lowest")
        else:
            self.addminperiod(self.p.period)

    def next(self):
        if not hasattr(self.data, "to_series"):
            self.lines.lowest[0] = min(self.data.get(size=self.p.period))


class CrossUp(bt.Indicator):
    lines = ("crossover",)

    def __init__(self, *args):
        d0, d1 = args if args else (self.data0, self.data1)
        if hasattr(d0, "to_series"):
            v0 = d0.to_series()
            v1 = d1.to_series()
            diff = v0 - v1
            signs = np.sign(diff)
            prev = pd.Series(np.where(signs == 0, np.nan, signs)).ffill().shift(1)
            values = ((signs > 0) & (prev <= 0)).astype(float)
            self.lines.crossover = d0.wrap_indicator(values, name="crossover")
        else:
            self.data0 = d0
            self.data1 = d1
            self.addminperiod(2)

    def next(self):
        if not hasattr(self.data0, "to_series"):
            crossed = self.data0[0] > self.data1[0] and self.data0[-1] <= self.data1[-1]
            self.lines.crossover[0] = float(crossed)


class CrossDown(CrossUp):
    def __init__(self, *args):
        d0, d1 = args if args else (self.data0, self.data1)
        if hasattr(d0, "to_series"):
            v0 = d0.to_series()
            v1 = d1.to_series()
            diff = v0 - v1
            signs = np.sign(diff)
            prev = pd.Series(np.where(signs == 0, np.nan, signs)).ffill().shift(1)
            values = ((signs < 0) & (prev >= 0)).astype(float)
            self.lines.crossover = d0.wrap_indicator(values, name="crossover")
        else:
            self.data0 = d0
            self.data1 = d1
            self.addminperiod(2)

    def next(self):
        if not hasattr(self.data0, "to_series"):
            crossed = self.data0[0] < self.data1[0] and self.data0[-1] >= self.data1[-1]
            self.lines.crossover[0] = float(crossed)


class ATR(bt.Indicator):
    lines = ("atr",)
    params = (("period", 14),)

    def __init__(self):
        if hasattr(self.data.close, "to_series"):
            high = self.data.high.to_series()
            low = self.data.low.to_series()
            close = self.data.close.to_series()
            tr = pd.concat(
                [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                axis=1,
            ).max(axis=1)
            values = tr.ewm(alpha=1 / self.p.period, adjust=False).mean()
            self.lines.atr = self.data.close.wrap_indicator(values, name="atr")
        else:
            self.addminperiod(self.p.period + 1)

    def next(self):
        if not hasattr(self.data.close, "to_series"):
            ranges = []
            for i in range(self.p.period):
                high = self.data.high[-i]
                low = self.data.low[-i]
                prev_close = self.data.close[-i - 1]
                ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            self.lines.atr[0] = sum(ranges) / len(ranges)

class DonchianChannels(bt.Indicator):
    '''
    Donchian Channels
    '''
    alias = ('DCH', 'DonchianChannel',)
    lines = ('dcm', 'dch', 'dcl',)  # dc middle, dc high, dc low
    params = (
        ('period', 20),  # default value (could be modified)
        ('lookback', -1),  # consider current bar or not
    )

    def __init__(self):
        hi, lo = self.data.high, self.data.low
        if self.p.lookback:  # move backwards as needed
            hi, lo = hi(self.p.lookback), lo(self.p.lookback)

        self.l.dch = Highest(hi, period=self.p.period)
        self.l.dcl = Lowest(lo, period=self.p.period)
        self.l.dcm = (self.l.dch + self.l.dcl) / 2.0  # avg of the above


class TurtleSizer(bt.Sizer):
    params = (
        ("theta", 0.01),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        # 简化版 Sizing 逻辑，仅保留 ATR 约束部分用于演示兼容性
        atr = self.strategy.atr[data][0]
        abs_vol = atr * comminfo.p.mult
        # 假设起始资金或当前价值计算
        current_value = self.broker.get_value()
        atr_size = (current_value * self.p.theta) // abs_vol if abs_vol != 0 else 0
        return atr_size


class Turtle(bt.Strategy):
    """Turtle trading system"""

    params = (
        ("s1_longperiod", 20),
        ("s1_shortperiod", 10),
        ("s2_longperiod", 55),
        ("s2_shortperiod", 20),
        ("user_s1", True),
        ("bigfloat", 6),
        ("drawback", 1),
        ("closeout", 2),
        ("contract_max", 3),
        ("mkt_max", 6),
        ("dir_max", 12),
        ("theta", 0.01),
    )

    def __init__(self):
        # s1 or s2
        if self.p.user_s1:
            longperiod = self.p.s1_longperiod
            shortperiod = self.p.s1_shortperiod
        else:
            longperiod = self.p.s2_longperiod
            shortperiod = self.p.s2_shortperiod

        self.order = {}
        self.buyprice = {}
        self.sellprice = {}
        self.pos_count = {}
        
        self.atr = {}
        self.longsig = {}
        self.shortsig = {}
        self.longexit = {}
        self.shortexit = {}

        for d in self.datas:
            # 突破通道
            long_channel = DonchianChannels(d, period=longperiod)
            short_channel = DonchianChannels(d, period=shortperiod)

            # 交易信号 - 使用原版语法
            self.longsig[d] = CrossUp(d.close(0), long_channel.dch)
            self.shortsig[d] = CrossDown(d.close(0), long_channel.dcl)
            self.longexit[d] = CrossDown(d.close(0), short_channel.dcl)
            self.shortexit[d] = CrossUp(d.close(0), short_channel.dch)

            # atr
            self.atr[d] = ATR(d, period=20)
            
            self.order[d] = None
            self.buyprice[d] = None
            self.sellprice[d] = None
            self.pos_count[d] = 0

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.buyprice[order.data] = order.executed.price
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.sellprice[order.data] = order.executed.price

        self.order[order.data] = None

    def next(self):
        for d in self.datas:
            pos = self.getposition(d).size

            if not pos:
                if self.longsig[d]:
                    self.order[d] = self.buy(data=d)
                    self.pos_count[d] += 1
                elif self.shortsig[d]:
                    self.order[d] = self.sell(data=d)
                    self.pos_count[d] += 1
            else:
                if pos > 0:
                    # 多头平仓
                    if self.longexit[d] > 0:
                        self.order[d] = self.close(data=d)
                        self.pos_count[d] = 0
                else:
                    # 空头平仓
                    if self.shortexit[d] > 0:
                        self.order[d] = self.close(data=d)
                        self.pos_count[d] = 0
