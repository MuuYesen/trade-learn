# -*- coding: UTF-8 -*-
# Adapted from reference/backtrader-master/samples/order-execution/order-execution.py
# Only imports and class name changed to prove compatibility.

from importlib import import_module

import tradelearn.engine as bt


def _bt_indicators():
    return import_module("backtrader.indicators")


class SMA(bt.Indicator):
    lines = ("sma",)
    params = (("period", 30),)

    def __init__(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        if hasattr(line, "to_series"):
            self.lines.sma = bt.talib.SMA(line, timeperiod=self.p.period)
        else:
            native = _bt_indicators().SMA(line, period=self.p.period)
            self.lines.sma = native.lines[0]


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
            native = _bt_indicators().CrossOver(d0, d1)
            self.lines.crossover = native.lines[0]


class OrderExecutionStrategy(bt.Strategy):
    params = (
        ('smaperiod', 15),
        ('exectype', 'Market'),
        ('perc1', 3),
        ('perc2', 1),
        ('valid', 4),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        # print('%s, %s' % (dt.isoformat(), txt)) # Muted for benchmarking

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED', dt=self.data.datetime[0])
            self.order = order
            return

        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')

        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )

            else:  # Sell
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )

        # Sentinel to None: new orders allowed
        self.order = None

    def __init__(self):
        # SimpleMovingAverage on main data
        sma = SMA(period=self.p.smaperiod)

        # CrossOver (1: up, -1: down) close / sma
        self.buysell = CrossOver(self.data.close, sma)

        # Sentinel to None: new ordersa allowed
        self.order = None

    def next(self):
        if self.order:
            # An order is pending ... nothing can be done
            return

        # Check if we are in the market
        if self.position:
            # In the maerket - check if it's the time to sell
            if self.buysell[0] < 0:
                self.log(f"SELL CREATE, {self.data.close[0]:.2f}")
                self.sell()

        elif self.buysell[0] > 0:
            if self.p.exectype == 'Market':
                self.buy(exectype=bt.Order.Market)  # default if not given
                self.log(f"BUY CREATE, exectype Market, price {self.data.close[0]:.2f}")
