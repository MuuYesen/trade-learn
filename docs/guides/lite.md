# Lite Guide

Lite 是推荐起点：写法短，适合快速验证、教学、1.x 风格迁移和多资产目标权重。它不是独立撮合系统，而是同一套 `tradelearn.backtest` runtime 的轻量 facade。

## 基本结构

```python
import tradelearn.lite as tl


class MyStrategy(tl.Strategy):
    def init(self):
        self.ma = tl.tdx.MA(self.data.close, N=20)
        self.start_on_bar(21)

    def next(self):
        if self.data.close[0] > self.ma[0] and not self.position():
            self.buy(size=100)
        elif self.position():
            self.position().close()
```

## 数据访问

| 写法 | 含义 |
|---|---|
| `self.data.close[0]` | 当前 close |
| `self.data.close[-1]` | 上一根 close |
| `len(self.data)` | 当前推进到第几根 bar |
| `self.datas["AAPL"]` | 多标的按 symbol 访问 |
| `self.history_panel(20)` | 最近最多 20 根、所有标的的 OHLCV panel |

## 下单与持仓

```python
self.buy(size=100)
self.sell(size=100)
self.position().close()
self.order_target_percent(ticker="AAPL", target=0.5)
self.target_weights({"AAPL": 0.5, "MSFT": 0.3, "cash": 0.2})
self.close_all()
```

## 记录自定义数据

```python
self.record(alpha=score, signal=self.data.close[0])
```

运行后：

```python
backtest = tl.Backtest(data, MyStrategy)
stats = backtest.run()
stats.records
```
