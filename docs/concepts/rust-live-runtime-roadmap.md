# Rust Live Runtime 路线图

本文整理 trade-learn 后续向低延迟实盘演进时的目标、改动范围、预期结果和用户使用方式。

当前结论是：**底层可以逐步 Rust 化，但策略层应同时支持 Python 和 Rust**。Python 策略负责易用性和研究复用；Rust 策略负责真正低延迟热路径。两者共享同一套事件、订单意图、风控和执行语义。

## 背景

当前 `RustBroker` 的定位是高性能回测撮合内核，不是实盘交易网关。

它适合处理：

- 历史 bar / tick 的事件回放。
- 本地撮合、持仓、现金和权益更新。
- 与 `engine` / `lite` 回测结果对齐。

它不直接处理：

- 券商或交易所连接。
- 真实订单 ack / reject / partial fill。
- 撤改单竞态。
- 实盘行情订阅。
- 网络异常和交易通道重连。

因此，低延迟实盘不应该把当前 `RustBroker` 直接改成 live broker，而应该新增一条 Rust live runtime 链路。

## 目标

长期目标是形成统一的执行模型：

```text
MarketData -> Rust Runtime -> Strategy Adapter -> Risk -> Execution -> Order/Fills
```

其中：

- Rust runtime 负责事件循环、行情分发、风控、订单状态机、执行网关和异步日志。
- Python strategy adapter 允许用户继续用 Python 写策略。
- Rust strategy trait 允许高频用户用 Rust 写策略。
- 回测、仿真、实盘共用同一套事件和订单语义。

目标不是把所有用户都推向 Rust，而是让同一个系统支持两档能力：

| 策略类型 | 推荐语言 | 适用场景 |
|---|---|---|
| 研究、日频、分钟级、普通实盘 | Python | 快速迭代、pandas/numpy、报告、QMT 等普通执行链路 |
| tick 高频、毫秒级或更低延迟 | Rust | 常驻事件循环、低对象分配、低跨语言开销、低延迟执行 |

## 需要改动的模块

### 1. 统一事件模型

需要抽象出 runtime-neutral 的事件和数据结构。

建议位置：

```text
tradelearn/core/events.py
tradelearn/core/broker_contracts.py
tradelearn/core/market_data_contracts.py
```

核心对象包括：

```text
Bar
Tick
OrderIntent
OrderAck
OrderStatus
Fill
Position
Account
Timer
```

这些对象是 Python 策略、Rust 策略、回测 runtime 和实盘 adapter 之间的共同语言。

### 2. 调整 Python Strategy 语义

现有公开入口继续保留：

```python
from tradelearn.engine import Strategy
```

但策略内部应逐步减少对 backtest 内部对象的依赖，转向事件方法：

```python
class MyStrategy(Strategy):
    def on_start(self):
        ...

    def on_bar(self, bar):
        ...

    def on_tick(self, tick):
        ...

    def on_fill(self, fill):
        ...
```

`buy()`、`sell()`、`cancel()` 不直接绕过 runtime，而是生成标准化订单意图：

```text
Strategy -> OrderIntent -> Risk -> Execution
```

这样 Python 策略和 Rust 策略都走同一套风控与订单状态机。

### 3. 新增 Rust Runtime Core

新增或扩展 Rust 侧 runtime，而不是继续把 live 逻辑塞进 Python `backtest/engine.py`。

建议结构：

```text
tradelearn-rs/
  runtime/
  market_data/
  execution/
  risk/
  strategy/
  replay/
```

职责包括：

- 事件循环。
- 行情接收和分发。
- 策略调用。
- 本地同步风控。
- 订单状态机。
- 回测撮合 / 实盘执行 adapter。
- 异步日志和 telemetry。

### 4. PythonStrategyAdapter

Python 策略接入 Rust runtime 时，需要一个 adapter：

```text
Rust Runtime 收到事件
-> 转成 Python Bar/Tick/Fill
-> 调用 Python strategy.on_bar/on_tick/on_fill
-> 收集 Python 策略产生的 OrderIntent
-> 交回 Rust Risk/Execution
```

这条路径可以保持用户体验，但不适合极低延迟逐 tick 高频，因为每次事件都会跨 Python/Rust 边界。

### 5. Rust Strategy Trait

高频策略应支持 Rust 原生 trait：

```rust
pub trait Strategy {
    fn on_start(&mut self, ctx: &mut StrategyContext) {}
    fn on_tick(&mut self, tick: &Tick, ctx: &mut StrategyContext) {}
    fn on_bar(&mut self, bar: &Bar, ctx: &mut StrategyContext) {}
    fn on_fill(&mut self, fill: &Fill, ctx: &mut StrategyContext) {}
}
```

策略通过 `ctx.buy()`、`ctx.sell()`、`ctx.cancel()` 生成订单意图，不直接绕过风控。

初期可以支持静态编译进 binary。后续再考虑动态库 plugin 或 WASM。

### 6. Broker 与 MarketData Adapter

实盘 adapter 应挂在 runtime 边界，而不是散落在策略代码中。

执行 adapter：

```text
BacktestExecutionAdapter
QmtExecutionAdapter
CtpExecutionAdapter
FixExecutionAdapter
```

行情 adapter：

```text
CsvReplayMarketData
QmtMarketData
CtpMarketData
ExchangeMarketData
```

Rust runtime 只消费标准事件，不关心底层数据来自 CSV、QMT、CTP 还是交易所行情源。

## 分阶段结果

### 第一阶段：统一语义

目标：

- 明确 `Strategy`、`OrderIntent`、`Fill`、`Position`、`Account` 的公开/内部边界。
- 让 Python 策略更少依赖 `tradelearn.backtest` 内部对象。
- 保持现有 `engine` / `lite` 回测不破坏。

预期结果：

- 用户继续用 Python 写策略。
- 回测、普通实盘和部署侧 adapter 的订单语义更一致。
- 为 Rust runtime 接入打基础。

### 第二阶段：Rust Replay / Backtest Runtime

目标：

- Rust runtime 接管事件回放主循环。
- Python 策略通过 PythonStrategyAdapter 运行。
- 当前 Rust backtest 能力与新的事件模型对齐。

预期结果：

- Python 策略可以跑在 Rust runtime 上。
- 回测结果继续与原有 oracle 对齐。
- runtime 边界清晰，为 live adapter 做准备。

### 第三阶段：Rust Live Runtime

目标：

- Rust runtime 支持实盘行情、风控、执行和订单状态机。
- 支持 QMT / CTP / FIX 等执行 adapter。
- 日志、S3、监控全部异步化，避免阻塞交易线程。

预期结果：

- Python 策略可用于低频/中频实盘。
- Rust 策略可用于低延迟实盘。
- 研究、回测、仿真、实盘共享事件和订单语义。

### 第四阶段：Rust 策略生态

目标：

- 提供 Rust strategy trait。
- 提供 Rust 策略模板、示例和构建工具。
- 支持从 Python 研究结果导出 Rust 实盘配置。

预期结果：

- 用户可以选择 Python 策略或 Rust 策略。
- 高频用户不需要走 Python 热路径。
- trade-learn 形成双语言策略生态。

## 用户使用方式

### Python 策略

适合研究、日频、分钟级和普通实盘。

示例：

```python
from tradelearn.engine import Strategy


class MeanReversionStrategy(Strategy):
    def on_bar(self, bar):
        position = self.position(bar.symbol)

        if bar.close < self.indicator("lower_band", bar.symbol):
            self.buy(bar.symbol, qty=100)
        elif position.qty > 0 and bar.close > self.indicator("upper_band", bar.symbol):
            self.sell(bar.symbol, qty=position.qty)
```

运行方式：

```python
import tradelearn.engine as bt

engine = bt.Engine(runtime="rust")
engine.add_strategy(MeanReversionStrategy)
engine.run()
```

其中 `runtime="rust"` 表示底层事件循环、风控和执行由 Rust runtime 承担；策略回调仍在 Python 中执行。

### Rust 策略

适合 tick 高频、低延迟实盘和对延迟敏感的仿真。

示例：

```rust
use tradelearn_rs::prelude::*;

pub struct ImbalanceStrategy {
    threshold: f64,
}

impl Strategy for ImbalanceStrategy {
    fn on_tick(&mut self, tick: &Tick, ctx: &mut StrategyContext) {
        if tick.imbalance > self.threshold {
            ctx.buy(&tick.symbol, 100.0, OrderType::Market);
        }
    }

    fn on_fill(&mut self, fill: &Fill, ctx: &mut StrategyContext) {
        ctx.log_fill(fill);
    }
}
```

运行方式：

```bash
tradelearn-rs run \
  --strategy target/release/libimbalance_strategy.so \
  --config config.yaml
```

Rust 策略不经过 Python 热路径，适合真正低延迟场景。

### 共享配置

Python 研究和 Rust 实盘应共享同一份参数配置。

示例：

```yaml
strategy:
  name: alpha101_hs300
  version: v1
  symbols_source: hs300
  rebalance_every: 5

risk:
  max_position_pct: 0.05
  max_order_qty: 100000
  max_daily_turnover: 2.0

execution:
  adapter: qmt
  account_id: "40705983"
  order_type: market
```

这样可以避免 Python 回测参数和 Rust 实盘参数不一致。

## 不建议做的事

不要把当前 `RustBroker` 直接改成实盘 broker。

原因：

- 当前 `RustBroker` 是回测撮合和组合记账代理。
- 实盘需要处理网络、ack、reject、partial fill、撤改单竞态和重连。
- 两者职责不同，强行合并会让回测内核和实盘网关互相污染。

不要假设“broker Rust 化”等于极低延迟。

低延迟链路包含：

```text
行情接收 -> 行情解析 -> 状态更新 -> 策略决策 -> 风控 -> 订单生成 -> 执行网关 -> ack/fill
```

只把 broker 改成 Rust，只能优化其中一小段。如果行情、策略、风控和执行仍然经过 Python、pandas、HTTP 或 QMT 客户端，整体延迟仍会被这些环节限制。

## 设计原则

- **公开 API 稳定**：用户优先从 `tradelearn.engine` 和 `tradelearn.lite` 进入，不直接依赖 `tradelearn.backtest` 或 `tradelearn.core`。
- **事件语义统一**：Python 策略、Rust 策略、回测和实盘共享 `OrderIntent`、`Fill`、`Position` 等对象语义。
- **风控不可绕过**：策略只能提交订单意图，所有订单必须经过 runtime 风控。
- **低延迟热路径可 Rust 化**：真正高频场景应允许行情、策略、风控和执行都在 Rust 内完成。
- **Python 保持研究优势**：Python 继续负责投研、报告、配置生成、监控和低频实盘。

## 推荐路线

```text
1. 统一事件、订单意图和回报模型
2. 让 engine.Strategy 降低对 backtest 内部对象的依赖
3. Rust replay/backtest runtime 调用 Python 策略
4. 增加 Rust strategy trait
5. 增加 live execution / market data adapter
6. 完善异步日志、监控和 replay 验证
```

这条路线不要求推倒现有 `engine` / `lite`，而是在保持用户 API 稳定的前提下，把底层 runtime 逐步收敛到 Rust。
