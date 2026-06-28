# 扩展组件

trade-learn 的扩展原则是：**用户层写策略语法，research 层写投研组件，backtest 层执行事件回测，Rust 只承担高频内核**。日常扩展不需要改 Rust。

## 扩展点总览

| 扩展目标 | 推荐方式 | 入口 |
|---|---|---|
| Engine 专业策略组件 | `bt.Indicator` / `bt.Analyzer` / `bt.Sizer` / `bt.CommInfoBase` | `tradelearn.engine` |
| Lite 快速策略组件 | 普通函数 + `self.I(...)`，或 `target_weights()` 所需的权重 Series | `tradelearn.lite` |
| Research 预处理 / 特征 / 选股 / 权重 | sklearn-like 类：`fit()` / `transform()` / `fit_transform()` / `build()` | `tradelearn.research` |
| 数据源 | 返回标准 OHLCV DataFrame 或 `MultiIndex(timestamp, symbol)` panel | `tradelearn.data` |
| 报告 | 从 `Stats` 或 `Reporter` 读取结果生成 HTML / CSV / XLSX | `tradelearn.report` |
| paper / live broker | 实现 `tradelearn.core.Broker` 中性协议，通过事件回流成交状态 | 外部适配器 |

## Engine 高级拓展

Engine 是 Backtrader 风格入口。复杂策略组件应尽量写成 class，这样可复用、可测试，也更符合迁移用户心智。

### 自定义 Engine 指标

```python
import tradelearn.engine as bt


class MidPrice(bt.Indicator):
    lines = ("mid",)

    def __init__(self):
        self.lines.mid = (self.data.high + self.data.low) / 2.0


class AboveMid(bt.Strategy):
    def __init__(self):
        self.mid = MidPrice(self.data)

    def next(self):
        if not self.position and self.data.close[0] > self.mid.mid[0]:
            self.buy(size=10)
```

内置 vendor 指标仍然走统一命名空间：

```python
self.ma20 = bt.tdx.MA(self.data.close, N=20)
self.rsi14 = bt.talib.RSI(self.data.close, timeperiod=14)
```

### 自定义 Sizer

```python
class FixedCashSizer(bt.Sizer):
    params = (("cash_per_trade", 10_000),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        price = data.close[0]
        return int(self.p.cash_per_trade / price) if price > 0 else 0


cerebro.addsizer(FixedCashSizer, cash_per_trade=20_000)
```

### 自定义 Analyzer

```python
class TurnoverAnalyzer(bt.Analyzer):
    def start(self):
        self.turnover = 0.0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.turnover += abs(order.executed.size * order.executed.price)

    def get_analysis(self):
        return {"turnover": self.turnover}


cerebro.addanalyzer(TurnoverAnalyzer, _name="turnover")
```

### 自定义佣金

```python
class FixedPlusPercent(bt.CommInfoBase):
    params = (("commission", 0.0003), ("fixed", 1.0))

    def getcommission(self, size, price):
        return abs(size) * price * self.p.commission + self.p.fixed


cerebro.setcommission(comminfo=FixedPlusPercent())
```

## Lite 轻量拓展

Lite 的定位是快速表达策略。自定义指标优先写成普通函数，然后通过 `self.I(...)` 注册。

```python
import pandas as pd
import tradelearn.lite as tl


def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std


class MeanReversion(tl.Strategy):
    def init(self):
        self.z = self.I(zscore, self.data.close, window=20)
        self.start_on_bar(20)

    def next(self):
        if self.z[0] < -2:
            self.buy(size=100)
        elif self.position():
            self.position().close()
```

内置指标与 Engine 一样使用 vendor 命名空间：

```python
self.ma20 = tl.tdx.MA(self.data.close, N=20)
self.rsi14 = tl.talib.RSI(self.data.close, timeperiod=14)
self.tv_rsi = tl.tv.RSI(self.data.close, length=14)
```

## Research 通用组件拓展

Research 组件适合投研流水线：训练集 fit、测试集 transform、最后生成 scores 或 weights。

### 自定义 transformer

```python
import pandas as pd


class RankNormalizer:
    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, frame: pd.DataFrame):
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.columns] = out[self.columns].rank(pct=True)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
```

接入 `Pipeline`：

```python
pipe = research.Pipeline([RankNormalizer(columns=["alpha"])])
train = pipe.fit_transform(train)
test = pipe.transform(test)
```

### 自定义 Allocator

Allocator 的职责是把 scores 转成目标权重。最简单的扩展只需要实现 `build(scores)`。

```python
class LongOnlyTopN:
    def __init__(self, n: int, gross: float = 0.95):
        self.n = n
        self.gross = gross

    def build(self, scores):
        selected = scores.groupby(level="timestamp").nlargest(self.n)
        selected.index = selected.index.droplevel(0)
        return selected.groupby(level="timestamp").transform(
            lambda x: self.gross / len(x)
        ).rename("weight")
```

如果只是常见 top-k 等权组合，内置写法更直接：

```python
import tradelearn.research.portfolio as pf

allocator = pf.Allocator(
    select=pf.TopK(k=50),
    weight=pf.EqualWeight(gross=0.95),
    constrain=pf.Constraints(max_weight=0.05, normalize=True),
)
weights = allocator.build(scores)
```

## Data / Report 通用拓展

### 自定义数据源

自定义 provider 不需要继承框架类，只要返回规范数据：

- 单标的：index 为时间，columns 至少包含 `open/high/low/close/volume`。
- 多标的：`MultiIndex(timestamp, symbol)`，columns 同上。

```python
class MyProvider:
    def history_ohlc(self, symbols, start=None, end=None, freq="1d"):
        ...
        return bars
```

Engine 的 `cerebro.adddata(panel)` 会自动按 symbol 拆 feed；Lite 的 `Backtest(panel, Strategy)` 也会识别 panel。

### 自定义报告

`Stats` 是报告层的统一输入：

```python
from tradelearn.report import Reporter

reporter = Reporter(strategy.stats)
reporter.report("report.html")

summary = reporter.summary()
trades = reporter.trades()
equity = reporter.equity_curve()
```

如果用户已有收益率序列，也可以使用兼容 pyfolio 心智的入口：

```python
Reporter.from_returns(returns).report("returns_report.html")
```

## paper / live broker 拓展

实盘 broker 不复用 RustBroker 状态机。外部适配器实现 `core` 的中性协议，并通过 `BrokerEventPump` 把成交、撤单、拒单回流给策略。

```python
from tradelearn.core.broker_contracts import AccountSnapshot, OrderAck, OrderRequest


class MyLiveBroker:
    def place(self, req: OrderRequest) -> OrderAck:
        ...

    def cancel(self, broker_oid: str) -> None:
        ...

    def positions(self):
        ...

    def account(self) -> AccountSnapshot:
        ...

    def on_fill(self, callback):
        ...
```

约束：

- 策略发出的是订单意图，不假设立即成交。
- 成交、撤单、拒单、状态变化必须通过 broker 事件回流。
- QMT / IB / CTP 等具体适配器可以在外部包或私有扩展里实现，不进入 `core`。
