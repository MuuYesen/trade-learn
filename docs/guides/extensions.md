# 扩展组件

trade-learn 的扩展原则是：用户扩展写在用户层，框架协议写在 `core`，回测执行留在 `backtest` 和 Rust 内核。日常扩展不需要改 Rust。

## 扩展点总览

| 目标 | 推荐扩展方式 | 所在入口 |
|---|---|---|
| 自定义 Lite 指标 | 普通函数 + `self.I(...)` | `tradelearn.lite` |
| 自定义 Engine 指标 | 继承 `bt.Indicator` | `tradelearn.engine` |
| 自定义 Analyzer | 继承 `bt.Analyzer`，通过 `cerebro.addanalyzer(...)` 挂载 | `tradelearn.engine` |
| 自定义 Sizer | 继承 `bt.Sizer` | `tradelearn.engine` |
| 自定义报告 | 使用 `Reporter` 或从 `Stats` 读取数据自行生成 | `tradelearn.report` |
| 自定义研究步骤 | 写 transformer / selector / allocator，并记录到 `ResearchRun` | `tradelearn.research` |
| 自定义 broker | 实现 `tradelearn.core.Broker` 协议并通过事件回流 | paper / live 扩展 |
| 自定义数据源 | 返回标准 OHLCV DataFrame 或 `MultiIndex(timestamp, symbol)` panel | `tradelearn.data` |

## Lite 自定义指标

Lite 的自定义指标保持函数式。函数接收 Series / array，返回可按 bar 读取的序列。

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

    def next(self):
        if self.z[0] < -2:
            self.buy(size=100)
        elif self.position():
            self.position().close()
```

内置口径建议直接用 vendor 命名空间：

```python
self.ma20 = tl.tdx.MA(self.data.close, N=20)
self.rsi14 = tl.talib.RSI(self.data.close, timeperiod=14)
self.tv_rsi = tl.tv.RSI(self.data.close, length=14)
```

## Engine 自定义指标

Engine 保持 Backtrader 风格。复杂指标建议显式写成 `bt.Indicator`，这样可组合、可测试、可复用。

```python
import tradelearn.engine as bt


class MidPrice(bt.Indicator):
    lines = ("mid",)

    def __init__(self):
        self.lines.mid = (self.data.high + self.data.low) / 2


class Strategy(bt.Strategy):
    def __init__(self):
        self.mid = MidPrice(self.data)

    def next(self):
        if self.data.close[0] > self.mid.mid[0]:
            self.buy(size=10)
```

如果是内置函数口径，Engine 和 Lite 写法保持一致：

```python
self.ma20 = bt.tdx.MA(self.data.close, N=20)
self.rsi14 = bt.talib.RSI(self.data.close, timeperiod=14)
```

## Analyzer / Sizer

Engine 保留 Backtrader 心智。Analyzer 负责观察回测结果，不直接改 broker；Sizer 负责把策略意图转换成默认下单数量。

```python
import tradelearn.engine as bt


class OrderCount(bt.Analyzer):
    def start(self):
        self.count = 0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.count += 1

    def get_analysis(self):
        return {"completed_orders": self.count}


cerebro = bt.Cerebro()
cerebro.addanalyzer(OrderCount, _name="orders")
```

## Engine 组件扩展示例

Engine 的定位是专业事件驱动入口，扩展方式尽量贴近 Backtrader：指标写成 `bt.Indicator`，分析器写成 `bt.Analyzer`，下单数量逻辑写成 `bt.Sizer`。这些组件只依赖 Engine facade，不需要碰 `backtest` 或 Rust。

### 自定义指标

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
        if not self.position and self.data.close[0] > self.mid[0]:
            self.buy()
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

## Research 扩展

研究层扩展应保持 sklearn-like 心智：`fit` 学训练集状态，`transform` 应用到新数据，`fit_transform` 只是便捷组合。

```python
from tradelearn import research

scaler = research.preprocess.StandardScaler(columns=["alpha"])
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

allocator = research.portfolio.Allocator.topk_equal(k=50, gross=0.95)
weights = allocator.build(scores)
```

如果扩展只在一个研究流程里使用，可以保留为普通函数；如果需要跨训练集 / 测试集保存状态，就应写成 transformer 类。

## Broker 扩展

实盘 broker 不复用 RustBroker 状态机，而是实现 `core` 的中性协议，并通过 `BrokerEventPump` 把外部事件回流到框架：

```python
from tradelearn.core import AccountSnapshot, Broker, Fill, OrderAck, OrderRequest


class MyLiveBroker:
    def place(self, req: OrderRequest) -> OrderAck:
        ...

    def cancel(self, broker_oid: str) -> None:
        ...

    def positions(self):
        ...

    def account(self) -> AccountSnapshot:
        ...

    def on_fill(self, cb):
        ...
```

核心约束：

- 策略发出的是订单意图，不假设立即成交。
- 成交、撤单、拒单、状态变化必须通过 broker 事件回流。
- QMT / IB / CTP 等具体适配器可以在外部包或私有扩展里实现，不进入 `core`。
