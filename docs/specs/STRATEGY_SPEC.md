# STRATEGY SPEC

用户 API 规格——**严格逐字对齐 backtrader**。这是"无缝承接存量策略"的底线。

## 1. 设计哲学

- Strategy 保持纯净(策略逻辑与追踪/分析解耦)
- 所有扩展(追踪 / 分析 / 报告)通过 `Cerebro.addanalyzer(...)` 挂载
- 参数通过类属性 `params` 声明,运行时通过 `self.p.xxx` 访问
- 零全局状态

## 2. 类层次

```
tradelearn.backtest.Strategy          # 基类(严格 backtrader 风格)
├── tradelearn.ml.MLStrategy          # ML 策略基类
└── (用户子类)
```

## 3. Strategy 基类定义

```python
class Strategy:
    # --- 类级声明 ---
    params: tuple[tuple[str, Any], ...] = ()

    # --- 实例生命周期 hooks ---
    def __init__(self): ...         # 声明指标,准备状态
    def start(self): ...            # 回测开始(暖机期结束)
    def prenext(self): ...          # 暖机期每根 bar(可选,一般不用)
    def next(self): ...             # 主逻辑:每根 bar 触发
    def stop(self): ...             # 回测结束

    # --- 通知回调(可选) ---
    def notify_order(self, order): ...
    def notify_trade(self, trade): ...
    def notify_cashvalue(self, cash, value): ...  # 可选
    def notify_timer(self, timer, when, *args, **kwargs): ...  # 预留

    # --- 下单 API(backtrader 兼容) ---
    def buy(self, data=None, size=None, price=None, exectype=None, ...): ...
    def sell(self, data=None, size=None, price=None, exectype=None, ...): ...
    def close(self, data=None, ...): ...
    def cancel(self, order): ...

    # --- 组合下单 ---
    def order_target_size(self, data, target=0): ...
    def order_target_value(self, data, target=0): ...
    def order_target_percent(self, data, target=0): ...

    # --- 数据访问 ---
    self.data                       # 主数据(backtrader 的 self.datas[0])
    self.datas                      # 所有数据列表
    self.p                          # 参数访问(别名 self.params)
    self.position                   # 当前持仓(针对 self.data)
    self.getposition(data)           # 指定数据的持仓
    self.broker                      # broker 对象(cash/value/...)

    # --- 指标注册 ---
    # 无需显式 self.I(...),直接把 ta 函数返回值赋给 self.xxx 即可
    # 框架扫描 __init__ 里的所有 Line 对象,自动管理暖机期
```

## 4. 典型策略写法

### 4.1 最简

```python
from tradelearn.backtest import Strategy
from tradelearn import ta

class SmaCross(Strategy):
    params = (
        ('fast', 10),
        ('slow', 20),
    )

    def __init__(self):
        self.ma1 = ta.sma(self.data.close, period=self.p.fast)
        self.ma2 = ta.sma(self.data.close, period=self.p.slow)

    def next(self):
        if not self.position:
            if self.ma1[0] > self.ma2[0]:
                self.buy()
        elif self.ma1[0] < self.ma2[0]:
            self.close()
```

### 4.2 多资产

```python
class PairsTrading(Strategy):
    params = (('lookback', 60), ('z_entry', 2.0), ('z_exit', 0.5))

    def __init__(self):
        self.d0 = self.datas[0]
        self.d1 = self.datas[1]
        self.spread = self.d0.close - self.d1.close
        self.mean = ta.sma(self.spread, period=self.p.lookback)
        self.std = ta.stdev(self.spread, period=self.p.lookback)

    def next(self):
        z = (self.spread[0] - self.mean[0]) / self.std[0]
        if abs(z) > self.p.z_entry and not self.getposition(self.d0).size:
            if z > 0:
                self.sell(data=self.d0); self.buy(data=self.d1)
            else:
                self.buy(data=self.d0); self.sell(data=self.d1)
        elif abs(z) < self.p.z_exit:
            self.close(data=self.d0); self.close(data=self.d1)
```

### 4.3 带通知

```python
class LoggingStrategy(Strategy):
    def notify_order(self, order):
        if order.status == order.Completed:
            self.log(f"FILL {order.ordtype} {order.size}@{order.executed.price}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE pnl={trade.pnl:.2f}")

    def log(self, msg):
        print(f"[{self.data.datetime.date(0)}] {msg}")
```

反转持仓通知语义:单笔成交若从多头反转为空头或从空头反转为多头,订单与 fill
仍保持一笔,但 `notify_trade()` 必须先收到旧方向 close leg(`isclosed=True`,
记录已实现 PnL),再收到新方向 open leg(`isopen=True`,记录反转后的新仓位)。
这保证策略侧可以同时观察完整平仓结果和新仓位打开事件。

## 5. 索引规则(严格 backtrader)

| 表达式 | 含义 |
|---|---|
| `self.data.close[0]` | 当前 bar 收盘价 |
| `self.data.close[-1]` | 昨日收盘价(前一根) |
| `self.data.close[-5]` | 5 根前收盘价 |
| `self.data.high[0]` / `.low[0]` / `.open[0]` / `.volume[0]` | 对应字段 |
| `self.data.datetime[0]` | 当前 bar 时间 |
| `self.data.datetime.date(0)` | 当前 bar 日期(date 对象) |

**索引 0 = 当前**,不是 `[-1]`。**这是对齐 backtrader 的硬约定**。

### 指标同理

```python
self.ma1[0]        # 当前 MA 值
self.ma1[-1]       # 昨日 MA 值
```

## 6. Order 对象

```python
class Order:
    # 状态常量(对齐 backtrader)
    Submitted: int
    Accepted: int
    Partial: int
    Completed: int
    Canceled: int
    Expired: int
    Margin: int
    Rejected: int

    # 方向常量
    Buy: int
    Sell: int

    # 类型常量
    Market: int
    Limit: int
    Stop: int
    StopLimit: int
    Close: int          # 平仓

    # 字段
    status: int
    ordtype: int        # Buy / Sell
    exectype: int       # Market / Limit / Stop / ...
    size: float
    price: float | None
    executed: ExecutedInfo

class ExecutedInfo:
    size: float
    price: float
    value: float         # size × price
    comm: float          # commission
    pnl: float
```

## 7. Trade 对象

```python
class Trade:
    ref: int             # 交易编号
    data: Data           # 归属数据
    size: float          # 头寸大小(进入时)
    price: float         # 进入价
    value: float
    commission: float
    pnl: float           # 已实现盈亏
    pnlcomm: float       # 扣除佣金后
    isopen: bool
    isclosed: bool
    status: int          # Created / Open / Closed
    dtopen: datetime
    dtclose: datetime | None
```

## 8. 下单方法详解

### 8.1 buy / sell

```python
self.buy(
    data=None,            # None = self.data
    size=None,            # None = 用 getsizer()(默认 1 股)
    price=None,           # None = 市价
    exectype=None,        # None = Market
    valid=None,           # 订单有效期
    tradeid=0,
    oco=None,             # 1.0 不支持
    trailamount=None,     # 1.0 不支持
    ...
)
```

### 8.2 close

```python
self.close(data=None)    # 平当前持仓(size 自动)
```

### 8.3 order_target_*

```python
# 目标 size(绝对股数)
self.order_target_size(data, target=100)

# 目标市值
self.order_target_value(data, target=50000)

# 目标仓位比例(按 equity 计算)
self.order_target_percent(data, target=0.1)  # 10%
```

**框架自动计算差额并发出 buy/sell**。

## 9. Position 对象

```python
class Position:
    size: float          # 正 = 多头,负 = 空头,0 = 空仓
    price: float         # 平均成本
    adjbase: float       # 调整基准价

    def __bool__(self):
        return self.size != 0
```

### 使用

```python
if not self.position:              # 等价 if self.position.size == 0
    ...

if self.position.size > 0:
    self.log(f"long {self.position.size}@{self.position.price}")
```

## 10. Sizer(仓位计算器,1.0 简化)

backtrader 有完整 Sizer 体系,**1.0 我们简化**:

- `self.buy()` 不传 size → 默认 1 股
- 用户显式传 size
- 或用 `order_target_*` 自动计算

**不做**:`FixedSize / PercentSizer / StopTrailSizer` 等。用户可以用 `order_target_percent` 达到"百分比仓位"效果。

## 11. Analyzer 基类

```python
class Analyzer:
    params: tuple = ()
    strategy: Strategy = None          # 由 Cerebro 注入

    def __init__(self): ...

    def on_start(self): ...
    def on_bar(self, bar): ...
    def on_fill(self, fill): ...
    def on_trade(self, trade): ...
    def on_end(self, stats): ...

    def get_analysis(self) -> dict:
        """返回分析结果。"""
        raise NotImplementedError
```

### 内置 Analyzer(1.0)

| 类 | 用途 |
|---|---|
| `MLflowAnalyzer` | 上报 MLflow |
| `SharpeAnalyzer` | 快速 Sharpe(对应 backtrader bt.analyzers.SharpeRatio) |
| `DrawdownAnalyzer` | 回撤(对应 bt.analyzers.DrawDown) |
| `TradeAnalyzer` | 交易统计(对应 bt.analyzers.TradeAnalyzer) |
| `AnnualReturnAnalyzer` | 年化收益 |

## 12. 与 backtrader 的行为一致性

### 应严格一致

- `params` 声明方式
- `self.p.xxx` 访问
- `__init__` 生命周期
- `next / notify_order / notify_trade` 语义
- 索引规则(`[0]` 当前)
- Order / Trade / Position 字段与状态常量

### 允许略有差异(文档里说明)

- Order 枚举值的具体数字(不重要)
- Strategy 没有 `prenext / nextstart` 的双触发(我们只在暖机后触发 next)
- 没有 `Cerebro.optstrategy` 全套(grid_search 走 MLflow 层)
- 没有 `writer` 机制(用 Analyzer)

## 13. 与 backtesting.py 的差异(迁移提示)

| backtesting.py | backtrader / trade-learn |
|---|---|
| `init(self)` | `__init__(self)` |
| `self.I(func, ...)` 注册 | 直接赋值 `self.ma = ta.sma(...)` |
| `self.data.close[-1]` 当前 | `self.data.close[0]` 当前 |
| 类属性 `fast = 10` | `params = (('fast', 10),)` |
| `self.buy(size=100)` | `self.buy(size=100)`(签名对齐) |

## 14. 错误处理

```python
# 用户常见错误 + 友好提示

class StrategyError(TradelearnError):
    """Raised from Strategy."""

# 暖机期没到就用指标:
>>> RuntimeError: indicator 'ma1' not ready at bar 5 (period=10).
    It will be available from bar 10 onwards. Consider using `prenext`
    or guarding with `if len(self.ma1) > 0:`.

# 索引越界:
>>> IndexError: tried to access close[-20] but only 15 bars are available.

# 参数错误:
>>> ValueError: Strategy.params must be a tuple of (name, value) pairs,
    got <dict>.
```

## 15. 测试要求

- 每个公开方法至少 1 个 docstring Example(doctest)
- 10 个金标策略覆盖典型场景
- 与 backtrader 相同策略对照测试(`tests/compat/backtrader/`):
  - 10 个知名 backtrader 开源策略
  - 改 1 行 import,跑通 + 结果对齐

## 16. 不做的事(1.0)

- ❌ `Strategy.optstrategy`(参数优化,走外层 grid_search)
- ❌ 完整 Sizer 体系(简化为 `order_target_*`)
- ❌ `getdatabyname` 动态数据注入(静态 adddata)
- ❌ Writer / Observer / Plot_*(都归 Analyzer)
- ❌ `replaydata / resampledata` 完整实现(基础多 timeframe 支持)
