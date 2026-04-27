# BACKTEST SPEC

Rust 事件撮合核 + Python 门面的行为规格。**Clean-Room 实现参照 backtesting.py,API 对标 backtrader**。

## 1. 设计基准

- **内核语义**:参考 backtesting.py(事件驱动 / 订单状态机 / 成交价规则)
- **用户 API**:严格对标 backtrader(Cerebro / Strategy / Analyzer)
- **实现路径**:Clean-Room(阶段 2 Week 5 写设计笔记 → 冷冻 2 周 → Rust 独立实现)
- **合规**:Apache-2.0 主协议 + NOTICE "inspired by"

## 2. 事件类型

```rust
// backtest-rs/src/events.rs
pub enum Event {
    Bar(BarEvent),
    Order(OrderEvent),
    Fill(FillEvent),
    Cancel(CancelEvent),
    Reject(RejectEvent),
    Timer(TimerEvent),      // 预留(实盘定时任务)
    Tick(TickEvent),        // 预留(1.2+ tick 级)
}
```

### 字段(摘要)

- `BarEvent`:ts / symbol / ohlcv
- `OrderEvent`:order_id / symbol / side / type / size / limit / stop / created_ts
- `FillEvent`:order_id / symbol / size / price / commission / slippage / ts
- `CancelEvent`:order_id / reason
- `RejectEvent`:order_id / reason

## 3. 事件循环语义

### 3.1 单根 bar 处理顺序

```
1. 新 BarEvent 到达(历史回放 or 实时推送)
2. 更新当前 bar 时间 T
3. 处理上一 bar 遗留的 OrderEvents:
   a. 尝试撮合(按撮合规则)
   b. 产生 FillEvent / CancelEvent / RejectEvent
4. Mark-to-market:根据新 bar close 更新 equity / unrealized_pnl
5. 触发 Python callback: strategy.next()
   - Python 策略产生新订单 → 入队(等待下一 bar 撮合)
6. 触发 Analyzer.on_bar()(每个已注册 Analyzer)
7. 推进到下一 BarEvent
```

### 3.2 关键时点

| 时点 | 事件 |
|---|---|
| `T.open` | 昨日提交的订单在此撮合(默认) |
| `T.bar` | Python next() 触发 |
| `T.close` | Mark-to-market;可配置为撮合时点 |

**`trade_on_close` 参数**(Cerebro 层):
- `False`(默认):信号在 `next()` 产生 → 下根 bar 的 `open` 成交
- `True`:信号在 `next()` 产生 → 当根 bar 的 `close` 成交

### 3.3 首次启动

- 所有指标在 `Strategy.__init__()` 内声明
- 框架自动计算 `min_period`(所有指标中最长的 lookback)
- **前 `min_period` 根 bar 不触发 `next()`**(暖机期)
- 暖机结束后才开始触发策略逻辑

## 4. 订单类型(1.0)

### 4.1 Market

- 立即撮合(下一 bar 的 open 或当前 bar 的 close,按 `trade_on_close`)
- 成交价 = 目标价 + slippage

### 4.2 Limit

- 买单:下一 bar `low ≤ limit_price` 时成交,价格 = `min(limit_price, next_open)`
- 卖单:下一 bar `high ≥ limit_price` 时成交,价格 = `max(limit_price, next_open)`
- 未成交订单:`time_in_force` 控制生命周期(day 过期 / gtc 持续 / ioc 立即撤)

### 4.3 Stop

- 买单:`high ≥ stop_price` 时触发转 market
- 卖单:`low ≤ stop_price` 时触发转 market
- 触发后的成交价应用 slippage

### 4.4 Stop-Limit

- 触发条件同 Stop,触发后转 Limit 单(不是 market)

### 4.5 不做的(推 1.1+)

- ❌ OCO(One-Cancels-Other)
- ❌ Trailing Stop
- ❌ Iceberg(冰山)
- ❌ TWAP / VWAP

## 5. Slippage 模型

可插拔,通过 `Cerebro` 构造时指定:

```python
cerebro = Cerebro(slippage=FixedSlippage(0.01))
cerebro = Cerebro(slippage=PercentSlippage(0.001))
cerebro = Cerebro(slippage=BarRangeSlippage(ratio=0.5))
```

### 实现

| 模型 | 公式 |
|---|---|
| `FixedSlippage(amount)` | `price ± amount`(买+卖-) |
| `PercentSlippage(ratio)` | `price × (1 ± ratio)` |
| `BarRangeSlippage(ratio)` | `next_open + rand()×(next_high-next_low)×ratio`(买+卖-) |

**默认 `FixedSlippage(0.0)`**(无滑点,为了 reproducibility)。

## 6. Commission 模型

```python
cerebro = Cerebro(commission=FixedCommission(5))             # 5 元/笔
cerebro = Cerebro(commission=PercentCommission(0.0003))      # 万 3
cerebro = Cerebro(commission=TieredCommission(tiers=[...]))  # 分档
cerebro = Cerebro(commission=CNAStockCommission())            # A 股套装
```

### `TieredCommission`(分档费率)

`tiers` 使用 `(notional_threshold, ratio)` 列表表示成交额阈值与对应费率。
成交额为 `abs(size) * price`,实际费率取**不超过当前成交额的最高阈值**。
若没有任何阈值匹配,手续费为 0。

### `CNAStockCommission`(A 股专用预设)

内置:
- 买入:佣金 0.025%,最低 5 元
- 卖出:佣金 0.025% + 印花税 0.1%,最低 5 元
- 过户费:0.002%(沪市)

## 7. Portfolio 记账

### 7.1 核心量

```
equity     = cash + Σ(position_size × mark_price)
margin_used = Σ(|position_size × mark_price|) / leverage
unrealized_pnl = Σ((mark_price - avg_price) × position_size × direction)
realized_pnl   = 累积已平仓盈亏
```

### 7.2 Fee 处理

- 成交时立即从 `cash` 扣除 commission
- Slippage 通过成交价差已计入 pnl,不单独扣

### 7.3 多资产

- 每个 symbol 独立 position 记录
- equity / account 全局汇总
- 可通过 `cerebro.broker.getcash(symbol=None)` 查询全局现金

### 7.4 做空 / 保证金(1.0 简化)

- 支持做空(负 position)
- 保证金按 1:1(不加杠杆),即做空金额 = 现金扣除等额
- 不计算融券利息(推 1.1 实盘做)

## 8. Python API(严格 backtrader 风格)

### 8.1 Strategy

```python
from tradelearn.backtest import Strategy

class SmaCross(Strategy):
    params = (
        ('fast', 10),
        ('slow', 20),
    )

    def __init__(self):
        # 注册指标 — 由框架管理暖机期
        self.ma1 = ta.sma(self.data.close, period=self.p.fast)
        self.ma2 = ta.sma(self.data.close, period=self.p.slow)

    def next(self):
        # self.data.close[0]  当前 bar close
        # self.data.close[-1] 昨日 close
        # self.ma1[0]          当前 MA
        # self.position        当前持仓
        if not self.position:
            if self.ma1[0] > self.ma2[0]:
                self.buy()
        else:
            if self.ma1[0] < self.ma2[0]:
                self.close()

    def notify_order(self, order):
        """订单状态变化回调(可选)。"""

    def notify_trade(self, trade):
        """完整交易闭合回调(可选)。"""
```

### 8.2 Cerebro

```python
from tradelearn.backtest import Cerebro

cerebro = Cerebro()
cerebro.adddata(bars)                       # 主数据
cerebro.adddata(bars_5m, name='5min')        # 多数据源(多 timeframe / 多资产)
cerebro.addstrategy(SmaCross, fast=10, slow=30)
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(0.002)
cerebro.addanalyzer(MLflowAnalyzer, experiment="sma_goog")
cerebro.run()
cerebro.plot()                               # bokeh HTML 图
```

### 8.3 Cerebro 构造参数

```python
Cerebro(
    slippage: SlippageModel = FixedSlippage(0),
    commission: CommissionModel = FixedCommission(0),
    trade_on_close: bool = False,
    exactbars: bool = False,            # 保留所有历史 bar(不做 rolling window 优化)
    stdstats: bool = True,               # 自动 attach 默认 analyzers
)
```

### 8.4 Analyzer

```python
from tradelearn.backtest import Analyzer

class MyAnalyzer(Analyzer):
    params = (('threshold', 0.5),)

    def __init__(self):
        self.values = []

    def on_start(self):
        """回测开始前。"""

    def on_bar(self, bar):
        """每根 bar 触发。"""

    def on_fill(self, fill):
        """成交时触发。"""

    def on_end(self, stats):
        """回测结束,stats 是 Stats 对象。"""

    def get_analysis(self):
        """返回分析结果(dict 或 pd.DataFrame)。"""
        return {"values": self.values}
```

### 8.5 self.data 索引规则

| 表达式 | 含义 |
|---|---|
| `self.data.close[0]` | 当前 bar 收盘价 |
| `self.data.close[-1]` | 昨日收盘价 |
| `self.data.close[-5]` | 5 日前收盘价 |
| `self.data.close.get(ago=0, size=20)` | 过去 20 根 bar 的收盘价切片 |

**索引 0 = 当前**(backtrader 约定),不是 `[-1]`。

## 9. Batched Callback(性能关键)

### 9.1 问题

Rust → Python 每根 bar callback 的 PyO3 开销累积会压垮性能。

### 9.2 解决

Rust 侧**批量累积 N 根 bar** 后一次性 callback:

```python
cerebro = Cerebro(callback_batch=1)    # 默认逐 bar(最精确)
cerebro = Cerebro(callback_batch=100)  # 100 根一批(性能优化)
```

**N > 1 时订单延迟 N 根 bar 生效**——回测里可接受。

实盘模式强制 `callback_batch=1`。

## 10. 决策确定性

- Rust 侧使用 `BTreeMap`(非 `HashMap`)
- 事件队列按 `(timestamp, submit_order)` 稳定排序
- 浮点累积固定顺序
- 全局 seed:`cerebro = Cerebro(seed=42)`

**相同输入 + 相同 seed → bit-level 相同输出**。

## 11. Broker 接入点

### SimBroker(1.0 唯一实现)

- Rust 核内置
- 通过 `cerebro.broker.setcash / setcommission` 配置

### QMTBroker(1.1)

```python
from tradelearn.brokers import QMTBroker

cerebro = Cerebro(broker=QMTBroker(proxy_url="http://127.0.0.1:8000"))
cerebro.run(mode='live')
```

## 12. 模式切换

```python
cerebro.run(mode='backtest')   # 1.0 默认
cerebro.run(mode='paper')      # 1.1 可选(SimBroker + LiveFeed)
cerebro.run(mode='live')       # 1.1 必做(Real Broker + LiveFeed)
```

1.0 只实现 `backtest` 模式,但 API 预留。

## 13. 性能目标

| 场景 | 目标 |
|---|---|
| 单品种 10 年日线(~2500 bar) | < 50 ms(Python 策略) |
| 500 股组合 10 年 | < 5 s |
| 向量化快批模式(callback_batch=100) | 上述数字 ÷ 5 |

## 14. Rust 侧模块划分

```
backtest-rs/
├── Cargo.toml
├── engine/               # 事件循环 + 调度
│   ├── loop.rs
│   └── feed.rs
├── matcher/              # 订单撮合
│   ├── market.rs
│   ├── limit.rs
│   └── stop.rs
├── portfolio/            # 记账
│   ├── account.rs
│   ├── position.rs
│   └── pnl.rs
├── slippage/
├── commission/
├── events/
└── py_binding/           # PyO3 绑定
    └── lib.rs
```

## 15. 金标对照

### 15.1 决策层(trades)

- 时间 / 方向 / size **0 差异**
- 参照源:1.x 回测结果(reference/tradelearn_1x)

### 15.2 数值层

- Equity curve `rtol=1e-6`
- 最终 stats `rtol=1e-4`(浮点累积顺序差异可解释)

### 15.3 对照流程

```python
# tests/consistency/test_sma_cross_rust_vs_1x.py
def test_sma_rust_decisions_match_1x():
    from reference.tradelearn_1x.backtest import Backtest as OldBT
    from tradelearn.backtest import Cerebro

    old_stats = OldBT(data, SmaCross).run()

    cerebro = Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    new_stats = cerebro.run()

    assert_trades_identical(new_stats.trades, old_stats.trades)
    assert np.allclose(new_stats.equity, old_stats.equity, rtol=1e-6)
```

## 16. 不做的事(1.0)

- ❌ Tick 级事件(推 1.2)
- ❌ Multi-threading 撮合(单线程 + 多进程外层并行)
- ❌ OCO / Trailing / TWAP
- ❌ 复杂保证金模型(按 1:1 处理)
- ❌ 融券利息 / 股息复投(推 1.1 实盘)
- ❌ 期货 / 期权(1.0 只股票 + 加密)
