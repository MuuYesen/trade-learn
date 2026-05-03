# 契约与边界

本页描述 trade-learn 各模块之间的**数据契约对象**——所有模块（data / indicators / factor / report / engine / lite / metrics / brokers）只通过这些对象通信。理解契约，就理解了"哪些字段会被框架信任、哪些是用户责任"。

> 这是面向高级用户的参考文档。日常使用 [Lite](../guides/lite.md) / [Engine](../guides/engine.md) 不需要直接构造这些对象——`tradelearn.data` 和指标函数已经返回符合契约的结果。

## 1. Bars（行情）

**形态**：`pd.DataFrame`，索引为 `MultiIndex(timestamp, symbol)`。

| 列 | dtype | 必须 | 说明 |
|---|---|:---:|---|
| `open` / `high` / `low` / `close` | float64 | ✅ | OHLC |
| `volume` | float64 | ✅ | 成交量 |
| `vwap` | float64 | ⚪ | 成交均价 |
| `amount` | float64 | ⚪ | 成交额 |
| `adj_factor` | float64 | ⚪ | 复权因子 |

**`DataFrame.attrs` 元数据**：

```python
{
    "market":  "CN" | "US" | "HK" | "CRYPTO",
    "freq":    "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d" | "1w",
    "adjust":  "pre" | "post" | "none",   # 默认 "pre"
    "engine":  "tv" | "tdx" | ...,
    "source":  "<URL 或标识>",
}
```

**不变量**：

- 时间戳 tz-aware，全项目统一 **UTC**；展示层按市场转换
- 同一 `(symbol, freq)` 下时间戳唯一
- 涨跌停 / 停牌行**保留**（`volume=0` 或 `NaN`），不得删除
- OHLC 关系：`low ≤ min(open, close) ≤ max(open, close) ≤ high`

**单资产切片**：

```python
bars.xs("GOOG", level="symbol")
```

## 2. Factor（因子面板）

`pd.DataFrame`，与 Bars 同样的 `MultiIndex(timestamp, symbol)`，每列是一个因子值（`float64`，允许 `NaN`，不允许 `inf`）。

`attrs`：

```python
{
    "factor_type": "momentum" | "value" | "quality" | "volatility" | "custom",
    "horizon":     int,      # 因子预测期（几根 bar）
    "version":     str,      # 因子定义版本
}
```

## 3. Signal（交易信号）

`pd.DataFrame`，索引 `MultiIndex(timestamp, symbol)`。

| 列 | dtype | 必须 | 值域 | 说明 |
|---|---|:---:|---|---|
| `weight` | float64 | ✅ | `[-1, 1]` | 目标仓位权重 |
| `confidence` | float64 | ⚪ | `[0, 1]` | 置信度 |
| `reason` | str | ⚪ | — | 可读原因 |

语义：`weight = 0` 空仓；`weight > 0` 做多（`0.1` 表示 10% 资金）；`weight < 0` 做空；`|weight| ≤ 1`，组合内总和 ≤ 1。

## 4. Returns（收益序列）

`pd.Series`，`DatetimeIndex(tz-aware UTC)`，`float64`。

- **简单收益** `r_t = (p_t / p_{t-1}) - 1`
- 对数收益**不作为契约形态**，需在入口转换

`attrs`：

```python
{
    "periods_per_year":  252 | 52 | 12 | ...,
    "rf":                float,    # 年化无风险利率，默认 0
    "benchmark_symbol":  str,      # 可选
}
```

## 5. StreamBar（流式单根 bar）

```python
@dataclass(frozen=True)
class StreamBar:
    ts: pd.Timestamp     # tz-aware UTC
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
```

用于 Rust 核回调 Python `Strategy.next()` 时传递单根 bar，以及实盘 tick / bar 推送的统一封装。

## 6. Stats（回测结果）

`cerebro.run()` 的返回值，不是独立契约对象，结构如下：

```python
@dataclass
class Stats:
    returns:   Returns               # 每日收益序列
    equity:    pd.Series             # 权益曲线
    trades:    pd.DataFrame          # 逐笔交易
    fills:     pd.DataFrame          # 逐笔成交
    positions: pd.DataFrame          # 每日持仓快照
    orders:    pd.DataFrame          # 订单历史
    summary:   dict[str, float]      # sharpe / max_dd / ... 关键指标
    analyzers: dict[str, Any]        # 各 Analyzer 产出
    config:    dict                  # 回测配置快照
```

详见 [Stats 结果对象](../concepts/stats.md)。

## 7. Broker Protocol

```python
class Broker(Protocol):
    # 下单
    def place(self, order: Order) -> OrderId: ...
    def cancel(self, oid: OrderId) -> None: ...
    def modify(self, oid: OrderId, **kwargs) -> None: ...

    # 查询
    def positions(self) -> list[Position]: ...
    def account(self) -> Account: ...
    def order_status(self, oid: OrderId) -> OrderStatus: ...

    # 异步回调
    def on_fill(self, cb: Callable[[Fill], None]) -> None: ...
    def on_cancel(self, cb: Callable[[OrderId], None]) -> None: ...
    def on_reject(self, cb: Callable[[OrderId, str], None]) -> None: ...

    # 生命周期
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
```

**1.0 仅内置 `SimBroker`**（Rust 核内部回测撮合）；`QMTBroker` 等实盘适配在 1.1 版本提供，必须实现 `Broker` Protocol + `BrokerEventPump`。

辅助类型：

```python
@dataclass(frozen=True)
class Order:
    symbol: str
    side: "buy" | "sell"
    type: "market" | "limit" | "stop" | "stop_limit"
    size: float
    limit: float | None = None
    stop: float | None = None
    time_in_force: "day" | "gtc" | "ioc" = "day"

@dataclass(frozen=True)
class Fill:
    order_id: OrderId
    symbol: str
    side: str
    size: float
    price: float
    commission: float
    ts: pd.Timestamp

@dataclass
class Position:
    symbol: str
    size: float
    avg_price: float
    market_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class Account:
    cash: float
    equity: float
    margin_used: float
    leverage: float

OrderStatus = Literal[
    "pending", "submitted", "accepted",
    "filled", "partially_filled",
    "cancelled", "rejected", "expired",
]
```

## 8. DataFeed Protocol（1.1 实盘预留）

```python
class DataFeed(Protocol):
    def subscribe(self, symbols: list[str], freq: str) -> None: ...
    def on_bar(self, cb: Callable[[StreamBar], None]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

实现：`HistoricalFeed`（回测，1.0）、`LiveFeed`（实盘，1.1）。

## 9. 验证函数（严入宽出）

每个契约对应一个 `validate_xxx` 函数，**只在模块边界做一次校验**，模块内部输出不再验证（信任自家代码）：

```python
from tradelearn.core import validate_bars

bars = validate_bars(df)   # 校验 MultiIndex、必填列、tz-awareness，违反即抛错
```

## 10. 类型落地位置

```
tradelearn/core/
├── bars.py          # validate_bars
├── factor.py        # validate_factor
├── signal.py        # validate_signal
├── returns.py       # validate_returns
├── events.py        # StreamBar / OrderEvent / FillEvent
├── broker.py        # Broker / DataFeed Protocol + Order / Fill / Position / Account
└── types.py         # Literal / TypeAlias 集中（OrderId / OrderStatus）
```

`tradelearn.core` 不依赖任何上层模块，是依赖图最底层。

## 11. 契约稳定性

1.0 发版后，任何字段、dtype、索引结构变化都视为 **breaking change**，需走主版本升级流程，并在 [v1 → v2 迁移](migration.md) 记录。在 1.0 之前，契约可能调整，但每次都会同步更新本页。

## 相关阅读

- [事件循环](event-loop.md)：Bars / Fill / Order 在引擎里如何被消费
- [撮合与成交](matching.md)：Order 进入 Broker 后的语义
- [Portfolio 模型](portfolio.md)：Fill 如何变成 Position 与 Equity
