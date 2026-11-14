# 快速开始

## 安装与环境

直接安装：

```bash
pip install trade-learn==0.3.5
```

下面的快速示例使用 TradingView 数据源、内置 TDX 指标及报告；如果要直接运行示例，请安装对应 extras：

```bash
pip install "trade-learn[research]==0.3.5"
```

推荐使用虚拟环境（如 `uv`）：

```bash
uv pip install trade-learn==0.3.5
# 或者克隆源码开发模式
uv pip install ".[research]"
```

0.3.5的统一`uv.lock`已通过检查，Store0.2.1a8已公开。PyPI wheel无需本机Rust工具链；只有源码开发才执行下面的maturin步骤。官方NA/Store组合需要CP313/314，见[执行接入](guides/live.md)。

如果需要开发 Rust 扩展：

```bash
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
        self.addminperiod(self.slow + 2)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position():
            self.buy(size=100)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position():
            self.position().close()


provider = TradingViewProvider(n_bars=5000)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()

print(stats.summary)
bt.plot()
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
        self.fast = bt.talib.SMA(self.data.close, timeperiod=self.p.fast)
        self.slow = bt.talib.SMA(self.data.close, timeperiod=self.p.slow)

    def next(self):
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=100)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


provider = TradingViewProvider(n_bars=5000)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro()
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.0003)
cerebro.broker.set_coc(True)
cerebro.adddata(bt.feeds.PandasData(dataname=bars), name="AAPL")
cerebro.addstrategy(SmaCross)

[strategy] = cerebro.run()

print(strategy.stats.summary)
from tradelearn.report import Reporter
Reporter(strategy.stats).report("report.html")
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

Provider 可以直接返回 `MultiIndex(timestamp, symbol)` 的 panel。Engine 按标的显式装配原生 Feed：

```python
symbols = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOG"]
bars = provider.history_ohlc(symbols, start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro()
for symbol in bars.index.get_level_values("symbol").unique():
    cerebro.adddata(bt.feeds.PandasData(dataname=bars.xs(symbol, level="symbol")), name=symbol)
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
tdx.MA(self.data.close, N=20)       # Engine
```

用户自定义指标按入口分层：

| 场景 | 推荐写法 |
|---|---|
| Lite 自定义指标 | 共同`Indicator`状态推进，批量/流式资格一致 |
| Engine 自定义复杂指标 | `class MyIndicator(bt.Indicator)` |

## 下一步

| 目标 | 阅读 |
|---|---|
| 写更完整的 Lite 策略 | [Lite 指南](guides/lite.md) |
| 写 Backtrader 风格策略 | [Engine 指南](guides/engine.md) |
| 理解 Stats 返回对象 | [Stats 结果对象](concepts/stats.md) |
| 做因子和机器学习研究 | [研究指南](guides/research.md) |
| 查看性能和对齐结果 | [性能基线](benchmarks.md) |

搜索、完整trial记录与有效配置隔离见[优化指南](guides/optimization.md)；已拟合bundle推理和当前评分边界见[研究指南](guides/research.md)。示例中的Provider请求需要相应数据授权；也可直接使用[本地Dataset](guides/data.md)，框架不会为本地查询自动取数。
