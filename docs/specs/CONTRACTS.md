# CONTRACTS

7 个核心契约对象的字段、索引、dtype、不变量规范。

所有 trade-learn 模块只通过这些契约对象通信。契约变更 = breaking change。

## 1. Bars(行情数据)

### 形态

```python
pd.DataFrame
```

### 索引

```
MultiIndex
  level 0: timestamp  (pd.DatetimeIndex, tz-aware UTC)
  level 1: symbol     (str)
```

### 列

| 列名 | dtype | 必须 | 说明 |
|---|---|:---:|---|
| open | float64 | ✅ | 开盘价 |
| high | float64 | ✅ | 最高价 |
| low | float64 | ✅ | 最低价 |
| close | float64 | ✅ | 收盘价 |
| volume | float64 | ✅ | 成交量 |
| vwap | float64 | ⚪ | 成交均价 |
| amount | float64 | ⚪ | 成交额 |
| adj_factor | float64 | ⚪ | 复权因子 |

### Attrs(DataFrame.attrs)

```python
{
    "market":  Literal["CN", "US", "HK", "CRYPTO"],
    "freq":    Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
    "adjust":  Literal["pre", "post", "none"],    # 默认 "pre"
    "engine":  str,                                # "tv" / "tdx" / ...
    "source":  str,                                # URL 或标识
}
```

### 不变量

- 时间戳必须 tz-aware,全项目统一 UTC(展示层按市场转换)
- 同一 `symbol` 在 `freq` 频率下时间戳唯一(无重复)
- 涨跌停 / 停牌时,行必须保留(`volume=0` 或 NaN,不得删除)
- OHLC 关系:`low ≤ min(open, close) ≤ max(open, close) ≤ high`

### 单资产便捷访问

```python
# 访问某 symbol 的切片
bars.xs('GOOG', level='symbol')          # → 单层时间索引 DataFrame
bars.loc[(slice(None), 'GOOG'), :]
```

## 2. Factor(因子面板)

### 形态

```python
pd.DataFrame
```

### 索引

```
MultiIndex
  level 0: timestamp
  level 1: symbol
```

### 列

每列是一个因子值:

| 列名 | dtype | 说明 |
|---|---|---|
| `<factor_name_1>` | float64 | NaN 允许 |
| `<factor_name_2>` | float64 | |
| ... | | |

### Attrs

```python
{
    "factor_type": Literal["momentum", "value", "quality", "volatility", "custom"],
    "horizon":     int,       # 因子预测 horizon(几根 bar)
    "version":     str,       # 因子定义版本
}
```

### 不变量

- 与 Bars 的 `(timestamp, symbol)` 索引对齐
- 单个因子允许全段 NaN(未来数据)
- 不允许 inf / -inf

## 3. Signal(交易信号)

### 形态

```python
pd.DataFrame
```

### 索引

```
MultiIndex(timestamp, symbol)
```

### 列

| 列名 | dtype | 必须 | 值域 | 说明 |
|---|---|:---:|---|---|
| weight | float64 | ✅ | [-1, 1] | 目标仓位权重 |
| confidence | float64 | ⚪ | [0, 1] | 置信度 |
| reason | str | ⚪ | — | 可读原因 |

### 语义

- `weight = 0` → 空仓
- `weight > 0` → 做多(0.1 = 10% 资金)
- `weight < 0` → 做空
- `|weight| ≤ 1` → 单资产不加杠杆;组合内和可 ≤ 1

## 4. Returns(收益序列)

### 形态

```python
pd.Series
```

### 索引

```
pd.DatetimeIndex(tz-aware UTC)
```

### 值

- dtype: `float64`
- **简单收益** `r_t = (p_t / p_{t-1}) - 1`
- 对数收益**不作为契约形态**,入口处转换

### Attrs

```python
{
    "periods_per_year":  int,      # 252 / 52 / 12 / ...
    "rf":                float,    # 无风险利率(年化),默认 0
    "benchmark_symbol":  str,      # 可选
}
```

### 不变量

- 时间戳升序
- NaN 允许(节假日 / 停牌),但不应有 inf

## 5. StreamBar(流式单根 bar)

### 形态

```python
@dataclass(frozen=True)
class StreamBar:
    ts: pd.Timestamp       # tz-aware UTC
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
```

### 用途

- Rust 核回调 Python `Strategy.next()` 时的单 bar 传递
- 未来流式 Indicator 的输入
- 实盘 tick/bar 推送的封装

### 不变量

- `frozen=True`(不可变)
- 字段完全对应 Bars 的列

## 6. Experiment(MLflow run 抽象)

### 形态

```python
@dataclass
class Experiment:
    name: str                       # experiment name
    run_name: str | None = None     # run name,None = 自动生成
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    run_id: str | None = None       # 由 MLflow 返回
    parent_run_id: str | None = None  # nested runs
```

### 用途

- `MLflowAnalyzer` 内部汇总
- 用户**不直接接触**这个对象
- 仅作为框架内部数据传递形式

## 7. Broker Protocol

### 形态

```python
from typing import Protocol, Callable

class Broker(Protocol):
    # 下单
    def place(self, order: Order) -> OrderId: ...
    def cancel(self, oid: OrderId) -> None: ...
    def modify(self, oid: OrderId, **kwargs) -> None: ...

    # 查询
    def positions(self) -> list[Position]: ...
    def account(self) -> Account: ...
    def order_status(self, oid: OrderId) -> OrderStatus: ...

    # 异步回调注册
    def on_fill(self, cb: Callable[[Fill], None]) -> None: ...
    def on_cancel(self, cb: Callable[[OrderId], None]) -> None: ...
    def on_reject(self, cb: Callable[[OrderId, str], None]) -> None: ...

    # 生命周期
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
```

### 实现(1.0 只有 SimBroker)

- `SimBroker`(Rust 核内部,回测默认)
- `QMTBroker`(1.1 版本,`brokers/qmt.py`)

### 辅助类型

```python
@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit", "stop", "stop_limit"]
    size: float               # 股数
    limit: float | None = None
    stop: float | None = None
    time_in_force: Literal["day", "gtc", "ioc"] = "day"

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

OrderId = str      # UUID
OrderStatus = Literal[
    "pending", "submitted", "accepted",
    "filled", "partially_filled",
    "cancelled", "rejected", "expired"
]
```

## 8. DataFeed Protocol(为 1.1 实盘预留)

```python
class DataFeed(Protocol):
    def subscribe(self, symbols: list[str], freq: str) -> None: ...
    def on_bar(self, cb: Callable[[StreamBar], None]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### 实现

- `HistoricalFeed`(回测,1.0)
- `LiveFeed`(实盘,1.1)

## 9. Stats(回测结果)

**不作为独立契约**,是 `cerebro.run()` 的返回值,结构:

```python
@dataclass
class Stats:
    returns: Returns              # 每日收益序列
    equity: pd.Series             # 权益曲线
    trades: pd.DataFrame          # 逐笔交易
    fills: pd.DataFrame           # 逐笔成交
    positions: pd.DataFrame       # 每日持仓快照
    orders: pd.DataFrame          # 订单历史
    summary: dict[str, float]     # 关键指标(sharpe/max_dd/...)
    analyzers: dict[str, Any]     # 各 Analyzer 的产出
    config: dict                  # 回测配置快照
```

## 10. 契约变更流程

1.0 前:

- 契约可自由调整,但每次必须更新本文档 + 影响到的 spec

1.0 后:

- 任何字段、dtype、索引结构变化 = **breaking change**
- 走 PR + 两人 review + MIGRATION.md 记录
- 版本号进入 2.0 或在 1.x 保持 deprecated 至少一个 minor

## 11. 类型定义落地位置

```
tradelearn/core/
├── __init__.py          # 对外暴露(Bars/Factor/Signal/... 不是类,是别名)
├── bars.py              # Bars validate 函数 + 便捷构造
├── factor.py            # Factor validate
├── signal.py            # Signal validate
├── returns.py           # Returns validate
├── events.py            # StreamBar / OrderEvent / FillEvent
├── experiment.py        # Experiment dataclass
├── broker.py            # Broker / DataFeed Protocol + Order / Fill / Position / Account
└── types.py             # Literal / TypeAlias 集中(OrderId / OrderStatus)
```

## 12. 验证函数约定

每个契约配一个 `validate_xxx` 函数,在模块边界做一次:

```python
# tradelearn/core/bars.py
def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Check MultiIndex, required columns, tz-awareness. Raise on violation."""
```

**严入宽出**:模块接收时验证,模块输出不验证(信任自家代码)。
