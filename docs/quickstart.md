# 快速开始

## 安装与环境

直接安装：

```bash
pip install trade-learn
```

推荐使用虚拟环境（如 `uv`）：

```bash
uv pip install trade-learn
# 或者克隆源码开发模式
uv sync
```

如果需要开发 Rust 扩展：

```bash
cd backtest-rs
maturin develop --release
```

## Lite：最短路径

```python
import tradelearn.lite as tl
from tradelearn.data import TradingViewProvider


class LiteSmaCross(tl.Strategy):
    fast = 10
    slow = 20

    def init(self):
        self.fast_ma = tl.tdx.MA(self.data.close, N=self.fast)
        self.slow_ma = tl.tdx.MA(self.data.close, N=self.slow)
        self.start_on_bar(self.slow + 1)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position():
            self.buy(size=100)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position():
            self.position().close()


provider = TradingViewProvider(n_bars=500)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()

print(stats.summary)
bt.plot("plot.html")
bt.report("report.html")
```

Lite 返回的 `stats` 是用户主对象：

```python
stats["final_value"]
stats.summary
stats.equity
stats.trades
stats.records
stats.strategy
stats.config
```

## Engine：Backtrader 风格

```python
import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider


class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 20))

    def __init__(self):
        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)
        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)

    def next(self):
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=100)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


provider = TradingViewProvider(n_bars=500)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro(trade_on_close=True)
cerebro.setcash(100_000)
cerebro.setcommission(0.0003)
cerebro.adddata(bars, name="AAPL")
cerebro.addstrategy(SmaCross)

[strategy] = cerebro.run()

print(strategy.stats.summary)
cerebro.plot("plot.html")
cerebro.report("report.html")
```

Engine 侧结果挂在策略实例上：

```python
stats = strategy.stats
stats.summary
stats.equity
stats.trades
stats.positions
stats.orders
stats.config
```

## 多标的数据

Provider 可以直接返回 `MultiIndex(timestamp, symbol)` 的 panel。Engine 的 `adddata()` 会自动按 symbol 拆分：

```python
symbols = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"]
bars = provider.history_ohlc(symbols, start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro()
cerebro.adddata(bars)  # 自动按 symbol 拆成多个 feed
```

Lite 同样可以直接接收 panel：

```python
bt = tl.Backtest(bars, MyPortfolioStrategy, cash=100_000)
stats = bt.run()
```

## 指标写法

内置 vendor 指标在 Lite 和 Engine 中保持一致：

```python
tl.tdx.MA(self.data.close, N=20)       # Lite
bt.tdx.MA(self.data.close, N=20)       # Engine
```

用户自定义指标按入口分层：

| 场景 | 推荐写法 |
|---|---|
| Lite 自定义向量函数 | `self.I(func, self.data.close, ...)` |
| Engine 自定义复杂指标 | `class MyIndicator(bt.Indicator)` |

## 下一步

| 目标 | 阅读 |
|---|---|
| 写更完整的 Lite 策略 | [Lite 指南](guides/lite.md) |
| 写 Backtrader 风格策略 | [Engine 指南](guides/engine.md) |
| 理解 Stats 返回对象 | [Stats 结果对象](concepts/stats.md) |
| 做因子和机器学习研究 | [研究指南](guides/research.md) |
| 查看性能和对齐结果 | [性能基线](benchmarks.md) |
