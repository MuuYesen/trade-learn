# 组件调用速查

本页只列当前代码中已经实现、适合作为用户入口的调用方式。规划中或兼容层内部接口不会放在这里。

## Lite / Engine 总览

Lite 和 Engine 共用 `tradelearn.backtest` runtime、Rust broker 和 `Stats`，区别只在用户语法：

| 能力 | Lite | Engine |
|---|---|---|
| 推荐导入 | `import tradelearn.lite as tl` | `import tradelearn.engine as bt` |
| 策略基类 | `tl.Strategy` | `bt.Strategy` |
| 运行器 | `tl.Backtest(data, Strategy)` | `bt.Cerebro()` |
| 初始化 hook | `init()` | `__init__()` |
| 主循环 | `next()` | `next()` |
| 数据访问 | `self.data.close[0]` | `self.data.close[0]` |
| 多标的数据 | `self.datas["AAPL"]` / ticker 参数 | `self.datas[0]` / `getdatabyname()` |
| 指标入口 | `tl.tdx` / `tl.talib` / `tl.tv` / `tl.pta` | `bt.tdx` / `bt.talib` / `bt.tv` / `bt.pta` |
| 自定义指标 | `self.I(func, ...)` | `class MyInd(bt.Indicator)` |
| 目标权重 | `target_weights()` / `target_equal()` | `target_weights()` / `order_target_percent()` |
| Analyzer / Sizer / Signal | 不暴露 | Backtrader 风格扩展入口 |
| 结果对象 | `stats = bt.run()` | `[strategy] = cerebro.run(); strategy.stats` |
| 绘图 / 报告 | `bt.plot()` / `bt.report()` | `cerebro.plot()` / `cerebro.report()` |

## 数据输入

### 单标的 OHLCV

```python
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

stats = tl.Backtest(bars, MyLiteStrategy).run()

cerebro = bt.Cerebro()
cerebro.adddata(bars, name="AAPL")
```

### 多标的 panel

`provider.history_ohlc(["AAPL", "MSFT"], ...)` 返回 `MultiIndex(timestamp, symbol)` panel 时，Engine 和 Lite 都会自动识别多标的数据形态。

```python
bars = provider.history_ohlc(["NASDAQ:AAPL", "NASDAQ:MSFT"], start="2023-01-01")

# Lite: 构造时直接传 panel
stats = tl.Backtest(bars, MyPortfolio).run()

# Engine: adddata 直接传 panel，会按 symbol 自动拆 feed
cerebro = bt.Cerebro()
cerebro.adddata(bars)
```

规则：

- 普通 OHLCV DataFrame：注册为单个 feed。
- MultiIndex panel 且 index 包含 `symbol` / `ticker` 层：按 symbol 自动拆成多个 feed。
- feed name 默认使用 symbol；也可以在单标的场景显式传 `name=`。

## Warm-up 与历史窗口

Lite:

```python
class S(tl.Strategy):
    def init(self):
        self.start_on_bar(20)
```

Engine:

```python
class S(bt.Strategy):
    def __init__(self):
        self.addminperiod(20)
```

`history_panel(lookback)` 是 tradelearn 增强接口，不是 Backtrader 原生接口。它返回当前已经可见的最近窗口：

```python
panel = self.history_panel(20)
# index: timestamp, symbol
# columns: open, high, low, close, volume, 以及额外小写列
```

第一次触发 `next()` 时只能返回 1 根；第 20 根后才可能返回完整 20 根。需要满窗口时请配合 `start_on_bar()` 或 `addminperiod()`。

## 下单与持仓

| 目标 | Lite | Engine |
|---|---|---|
| 买入 | `self.buy(size=100)` | `self.buy(size=100)` |
| 卖出 | `self.sell(size=100)` | `self.sell(size=100)` |
| 平仓 | `self.position().close()` | `self.close()` |
| 当前持仓 | `self.position()` | `self.position` / `self.getposition()` |
| 目标数量 | `self.order_target_size(target=100)` | `self.order_target_size(target=100)` |
| 目标比例 | `self.order_target_percent(target=0.6)` | `self.order_target_percent(target=0.6)` |
| 多标的目标权重 | `self.target_weights({"AAPL": 0.5, "MSFT": 0.4})` | `self.target_weights({"AAPL": 0.5, "MSFT": 0.4})` |

Lite 的 `position()` 是方法，Engine 的 `position` 是 Backtrader 风格属性。

A 股实盘或仿真需要整手买入、可卖数量限制时，可以通过 facade 创建约束对象再传给 `target_weights`：`tl.TargetOrderConstraints(buy_lot_size=100, sell_lot_size=100, max_sell_qty_by_symbol=...)`。

## 指标写法

内置指标口径在 Lite 和 Engine 中保持一致：

```python
self.ma20 = tl.tdx.MA(self.data.close, N=20)
self.rsi14 = tl.talib.RSI(self.data.close, timeperiod=14)
self.tv_rsi = tl.tv.RSI(self.data.close, length=14)
```

Engine 中只需要把 `tl` 换成 `bt`：

```python
self.ma20 = bt.tdx.MA(self.data.close, N=20)
self.rsi14 = bt.talib.RSI(self.data.close, timeperiod=14)
```

自定义指标按入口区分：

```python
# Lite: 函数式
self.z = self.I(zscore, self.data.close, window=20)

# Engine: Backtrader 风格 class
class MidPrice(bt.Indicator):
    lines = ("mid",)

    def __init__(self):
        self.lines.mid = (self.data.high + self.data.low) / 2
```

## Stats 结果对象

Lite:

```python
stats = bt.run()
stats["final_value"]
stats.summary
stats.equity
stats.trades
stats.records
stats.strategy
stats.config
```

Engine:

```python
[strategy] = cerebro.run()
stats = strategy.stats
stats.summary
stats.equity
stats.trades
stats.config
```

两边共享 `Stats` 字段：`summary`、`equity`、`returns`、`fills`、`trades`、`positions`、`orders`、`config`。Lite 额外提供 `records` 和 `strategy` 便捷入口。

## Research 组件

```python
import tradelearn.research as research
import tradelearn.research.preprocess as pp
import tradelearn.research.portfolio as pf

feature_set = research.FeatureSet({"alpha": lambda p: p.close.pct_change(20)})
features = feature_set.fit_transform(bars).dropna()

pipe = research.Pipeline(
    [
        pp.Winsorizer(columns=["alpha"]),
        pp.StandardScaler(columns=["alpha"]),
    ]
)
features = pipe.fit_transform(features)

weights = pf.Allocator(
    select=pf.TopK(k=50),
    weight=pf.EqualWeight(gross=0.95),
    constrain=pf.Constraints(max_weight=0.05, normalize=True),
).build(scores)
```

Research 组件遵循 sklearn-like 心智：需要训练集状态的组件实现 `fit()` / `transform()` / `fit_transform()`；只做无状态排序或权重计算的组件可以保留为轻量类或函数。
