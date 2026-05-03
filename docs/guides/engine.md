# Engine Guide

Engine 是 Backtrader 风格高级入口，适合复杂事件驱动策略、Analyzer、Observer、Sizer、Signal 和未来 paper/live adapter。

## 基本结构

```python
import tradelearn.engine as bt


class MyStrategy(bt.Strategy):
    params = (("period", 20),)

    def __init__(self):
        self.ma = bt.tdx.MA(self.data.close, N=self.p.period)

    def next(self):
        if self.ma[0] != self.ma[0]:
            return
        if not self.position and self.data.close[0] > self.ma[0]:
            self.buy(size=100)
        elif self.position:
            self.close()
```

## Cerebro

```python
cerebro = bt.Cerebro(trade_on_close=True)
cerebro.setcash(100_000)
cerebro.setcommission(0.0003)
cerebro.adddata(bars, name="AAPL")
cerebro.addstrategy(MyStrategy, period=20)
[strategy] = cerebro.run()
```

## 多数据

```python
bars = provider.history_ohlc(["NASDAQ:AAPL", "NASDAQ:MSFT"], start="2023-01-01")
cerebro.adddata(bars)  # MultiIndex panel 会自动按 symbol 拆 feed
```

策略内：

```python
for data in self.datas:
    price = data.close[0]
    self.order_target_percent(data=data, target=0.2)
```

## Engine 专属扩展

- `bt.Analyzer` / `bt.analyzers`
- `bt.Observer` / `bt.observers`
- `bt.Sizer`
- `bt.SignalStrategy`
- `bt.CommInfoBase`
- `bt.grid_search()`
