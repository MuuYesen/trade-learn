# -*- coding: UTF-8 -*-
# Adapted from reference/backtrader-master/samples/macd-settings/macd-settings.py
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
        if hasattr(line, "_runtime_function"):
            self.lines.sma = bt.talib.SMA(line, timeperiod=self.p.period)
        else:
            native = _bt_indicators().SMA(line, period=self.p.period)
            self.lines.sma = native.lines[0]


class ATR(bt.Indicator):
    lines = ("atr",)
    params = (("period", 14),)

    def __init__(self):
        if hasattr(self.data.close, "_runtime_function"):
            self.lines.atr = bt.talib.ATR(
                self.data.high,
                self.data.low,
                self.data.close,
                timeperiod=self.p.period,
            )
        else:
            native = _bt_indicators().ATR(self.data, period=self.p.period)
            self.lines.atr = native.lines[0]


class MACD(bt.Indicator):
    lines = ("macd", "signal", "histo")
    params = (("period_me1", 12), ("period_me2", 26), ("period_signal", 9))

    def __init__(self):
        line = self.data.close if hasattr(self.data, "close") else self.data
        if hasattr(line, "_runtime_function"):
            values = bt.talib.MACD(
                line,
                fastperiod=self.p.period_me1,
                slowperiod=self.p.period_me2,
                signalperiod=self.p.period_signal,
            )
            self.lines.macd = values.macd
            self.lines.signal = values.macdsignal
            self.lines.histo = values.macdhist
        else:
            native = _bt_indicators().MACD(
                line,
                period_me1=self.p.period_me1,
                period_me2=self.p.period_me2,
                period_signal=self.p.period_signal,
            )
            self.lines.macd = native.macd
            self.lines.signal = native.signal
            self.lines.histo = native.macd - native.signal


class CrossOver(bt.Indicator):
    lines = ("crossover",)

    def __init__(self, *args):
        self.data0, self.data1 = args if args else (self.data0, self.data1)
        self.addminperiod(2)

    def next(self):
        up = self.data0[0] > self.data1[0] and self.data0[-1] <= self.data1[-1]
        down = self.data0[0] < self.data1[0] and self.data0[-1] >= self.data1[-1]
        self.lines.crossover[0] = 1.0 if up else -1.0 if down else 0.0


class MacdTharp(bt.Strategy):
    '''
    This strategy is loosely based on some of the examples from the Van
    K. Tharp book: *Trade Your Way To Financial Freedom*. The logic:

      - Enter the market if:
        - The MACD.macd line crosses the MACD.signal line to the upside
        - The Simple Moving Average has a negative direction in the last x
          periods (actual value below value x periods ago)

     - Set a stop price x times the ATR value away from the close

     - If in the market:

       - Check if the current close has gone below the stop price. If yes,
         exit.
       - If not, update the stop price if the new stop price would be higher
         than the current
    '''

    params = (
        # Standard MACD Parameters
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
        ('atrperiod', 14),  # ATR Period (standard)
        ('atrdist', 3.0),   # ATR distance for stop price
        ('smaperiod', 30),  # SMA Period (pretty standard)
        ('dirperiod', 10),  # Lookback period to consider SMA trend direction
    )

    def notify_order(self, order):
        if order.status == order.Completed:
            pass

        if not order.alive():
            self.order = None  # indicate no order is pending

    def __init__(self):
        self.macd = MACD(self.data,
                         period_me1=self.p.macd1,
                         period_me2=self.p.macd2,
                         period_signal=self.p.macdsig)

        # Cross of macd.macd and macd.signal
        self.mcross = CrossOver(self.macd.macd, self.macd.signal)

        # To set the stop price
        self.atr = ATR(self.data, period=self.p.atrperiod)

        # Control market trend
        self.sma = SMA(self.data, period=self.p.smaperiod)
        self.smadir = self.sma - self.sma(-self.p.dirperiod)
        self.order = None

    def start(self):
        self.order = None  # sentinel to avoid operrations on pending order

    def next(self):
        if self.order:
            return  # pending order execution

        if not self.position:  # not in the market
            if self.mcross[0] > 0.0 and self.smadir < 0.0:
                self.order = self.buy()
                pdist = self.atr[0] * self.p.atrdist
                self.pstop = self.data.close[0] - pdist

        else:  # in the market
            pclose = self.data.close[0]
            pstop = self.pstop

            if pclose < pstop:
                self.close()  # stop met - get out
            else:
                pdist = self.atr[0] * self.p.atrdist
                # Update only if greater than
                self.pstop = max(pstop, pclose - pdist)
