# -*- coding: UTF-8 -*-
# Adapted from reference/backtesting-main/RSI_backtest/main.py
# Only imports are changed to prove compatibility.

import tradelearn.engine as bt
import datetime

class FurCommInfo(bt.CommInfoBase):
    """定义期货的交易手续费和佣金"""
    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("commission", 3.45 / 10_000),
        ("mult", 300),
        ("stamp_duty", 0.001),
        ("margin", 0.1),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission * self.p.mult


class EnhancedRSI(bt.Strategy):
    params = (
        ("period", 11),
        ("thold_l", 50),
        ("thold_s", 75),
        ("stop_limit", 0.02),
        ("target_percent", 0.15),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

    def __init__(self):
        # 保存收盘价、开盘价、日期
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime

        # 设置指标 - 使用原版语法
        self.rsi_s = bt.ind.RSI_SMA(self.datas[0], period=self.p.period, safediv=True)
        if len(self.datas) > 1:
            self.rsi_l = bt.ind.RSI_SMA(self.datas[1], period=self.p.period, safediv=True)
        else:
            self.rsi_l = bt.ind.RSI_SMA(self.datas[0], period=self.p.period * 3, safediv=True)
            
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.period)

        self.order = None
        self.buyprice = None
        self.sellprice = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"LONG EXECUTED @ {order.executed.price:.2f}")
                self.buyprice = order.executed.price
            elif order.issell():
                self.log(f"SHORT EXECUTED @ {order.executed.price:.2f}")
                self.sellprice = order.executed.price

        self.order = None

    def next(self):
        if self.order:
            return

        # 入场信号
        longsig = self.rsi_l[0] > self.p.thold_l and self.rsi_s[0] > self.p.thold_s
        shortsig = self.rsi_l[0] < 100 - self.p.thold_l and self.rsi_s[0] < 100 - self.p.thold_s

        if not self.position:
            if longsig:
                self.order = self.order_target_percent(target=self.p.target_percent)
            elif shortsig:
                self.order = self.order_target_percent(target=-self.p.target_percent)
        else:
            cur_pos = self.getposition(data=self.datas[0]).size
            if cur_pos > 0:
                pct_change = self.dataclose[0] / self.buyprice - 1
                closesig = pct_change < -self.p.stop_limit
            else:
                pct_change = self.dataclose[0] / self.sellprice - 1
                closesig = pct_change > self.p.stop_limit

            if closesig:
                self.order = self.close()
