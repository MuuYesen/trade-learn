# 事件循环

trade-learn 的回测引擎是**确定性、单事件、逐 bar 推进**的。本页解释一根 bar 进入引擎后到底发生了什么——包括订单何时撮合、`next()` 何时被调用、Analyzer 何时收到回调，以及多数据源（multi-data）下的对齐规则。

## 单根 bar 的生命周期

每根 bar 进入事件循环时，引擎严格按以下顺序执行：

1. **读取 bar，更新当前时间戳**
2. **撮合上一根 bar 的遗留订单**——产生 `Fill` / `Cancel` / `Reject` 事件
3. **mark-to-market**——用当前 bar 的 close 刷新所有持仓的市值与未实现 PnL
4. **暖机期判断**：若最长 lookback 尚未达到，跳过 5–6
5. **调用 Python `strategy.next()`**——用户在此读取行情、指标，调用 `self.buy()` / `self.sell()` / `self.close()`
6. **将 `next()` 中提交的新订单入队**——下一根 bar 开始才具备撮合资格（`trade_on_close=True` 例外，见下）
7. **调用当前 bar 的 Analyzer 回调**（`on_order` / `on_fill` / `on_trade` / `on_bar`）
8. **推进到下一根 bar**

> 关键点：用户策略看到当前 bar 时，引擎已经完成了订单处理与持仓刷新——`self.position`、`self.broker.getcash()` 反映的都是"含上根遗留订单成交后"的状态。

## `trade_on_close` 的意义

默认情况下，`next()` 中创建的市价单在**下一根 bar 的 open** 成交。

```python
cerebro = Cerebro(trade_on_close=False)   # 默认
```

设置 `trade_on_close=True` 时，当前 `next()` 创建的市价单可在**当前 bar close** 成交：

```python
cerebro = Cerebro(trade_on_close=True)
```

适合"用收盘价决策、用收盘价成交"的研究型回测，但隐含使用 close 作为决策依据。

## 暖机期（warmup）

暖机期由策略中已注册指标的最长 lookback 自动推导。在暖机期内：

- 引擎照常推进数据缓冲、撮合 / mark-to-market
- **不调用** `strategy.next()`，避免用户读取未就绪的指标

如果指标对象未自动暴露 lookback，可在 `Strategy.__init__()` 显式声明：

```python
class MyStrategy(Strategy):
    def __init__(self):
        self.ma = ta.sma(self.data.close, period=200)
        self.addminperiod(200)         # 显式暖机期
```

## 多数据源（multi-data）对齐

`datas[0]` 是 **primary feed**（主时钟）。每根 primary bar 触发一次完整的"撮合 → mark → next → analyzer"生命周期。其他 `datas[1..]`（secondary feed）按以下规则对齐：

- 在每个 primary timestamp，secondary feed 取自身 `timestamp ≤ primary timestamp` 的最新 bar（**latest-at-or-before**）
- 若 secondary 在当前 primary timestamp 前没有任何 bar，其 line cursor 保持未就绪——用户读取 `self.datas[1].close[0]` 会得到 `IndexError`，与普通越界访问行为一致

> 这条规则避免了"用最短数据截断主数据"和"缺失 bar 时窥视未来"两种典型错误。

## Analyzer 回调时点

Analyzer 在同一根 bar 内会同步收到所有订单状态变化：

| 回调 | 何时触发 |
|---|---|
| `on_order(order)` | 订单状态变化：submitted / accepted / completed / cancelled / rejected / expired |
| `on_fill(fill)` | 订单成交（仅成交事件，不污染统计） |
| `on_trade(trade)` | 一笔交易闭环（开 + 平） |
| `on_bar(bar)` | 当前 bar 的策略逻辑执行后 |

`on_fill` 与 `on_trade` 不接收非成交订单事件，避免被取消 / 拒单干扰盈亏统计。

## `callback_batch` 与批处理

默认 `callback_batch=1`：逐 bar 调用 `next()`。`callback_batch=N>1` 时，引擎按 N 根 bar 一批批量推进：

- 策略在当前 bar 创建的订单**延迟 N 根 primary bar 后**才具备撮合资格
- `trade_on_close=True` 在 `callback_batch=1` 下保持原有特殊语义；与 `N>1` 组合时遵循批处理延迟

> 该模式是性能优化路径，不改变 `next()` 可观察的事件顺序。绝大多数策略保持默认 `callback_batch=1` 即可。

## Python ↔ Rust 边界

| 责任 | 归属 |
|---|---|
| 事件循环、订单队列、撮合、mark-to-market、portfolio 记账 | Rust 核（`_core.so`） |
| 策略 `__init__` / `next` / `notify_*`、指标计算、Analyzer | Python |
| 数据传递 | Apache Arrow zero-copy |

Rust 不持有 Python 对象生命周期；Python 不直接操作 Rust ledger 内部状态。

## 回测与实盘统一（1.1 EventRunner）

1.1 起，回测和实盘共用同一个单事件 `EventRunner`：

```
HistoricalDriver / LiveDriver / PaperDriver
      │
      ▼
EventRunner.on_bar() / on_broker_event()
```

硬约束：

- **回测和实盘不分裂成两套策略执行模型**——`strategy.next()` 等用户 API 100% 兼容
- 历史批量优化只能放在 driver 层，**不得改变 `next()` 可见顺序**
- 实盘 broker 通过 `BrokerEventPump` 接入，不污染核心事件循环

## 相关阅读

- [撮合与成交](matching.md)：第 2 步"撮合遗留订单"的具体规则
- [Portfolio 模型](portfolio.md)：第 3 步"mark-to-market"的记账细节
- [契约与边界](contracts.md)：Bar / Order / Fill 的字段定义
