# 实盘协议

0.2.9通过公开 `engine.StrategyRuntime` 协调默认研究、事件回放和外部驱动 paper/live 的策略生命周期。接入方负责事件调度、行情、权威订单/账户事实和执行保护；Runtime 不启动后台循环。完整调用与限制见 [Runtime](../concepts/runtime.md#正式增量策略runtime)。

## 回测 broker 与实盘 broker 的区别

| 项目 | Rust 回测 broker | paper / live broker |
|---|---|---|
| 职责 | 本地撮合、订单推进、portfolio 记账 | 连接外部交易系统，回传订单状态和成交 |
| 状态来源 | Rust 内核内部状态 | 外部 broker / 交易柜台 |
| 成交语义 | 按 bar 和撮合模式确定 | 由真实市场和 broker 回报决定 |
| 策略假设 | `next()` 内发单，后续事件更新状态 | 同样发单，但不能假设立即成交 |
| 接口边界 | 内部 Rust 回测 broker | 私有适配器实现 broker-neutral 协议 |

因此，为了加快回测而优化 RustBroker，不会自动改变实盘 broker；但策略层必须只依赖共同的“意图 + 事件”语义。

trade-learn 不公开 `tradelearn.live` 包。具体券商适配器由独立扩展或部署项目提供。增量策略使用 `tradelearn.engine` 的公开 API；Lite 仍用于批量研究，不能因共享生命周期就推定 `Lite.I` 已支持完整实盘。

## 中性订单协议

实盘适配器应该接收中性订单请求，返回委托确认，并通过事件回流成交、撤单、拒单和状态。这里描述的是框架内部协议，不是用户策略应直接导入的公开 API；具体字段定义见 `docs/internals/contracts.md`。

| 类型 | 关键字段 | 用途 |
|---|---|---|
| `OrderRequest` | `symbol`, `side`, `qty`, `order_type`, `limit_price`, `stop_price`, `tif`, `client_oid` | 策略发出的 broker-neutral 下单请求 |
| `OrderAck` | `client_oid`, `broker_oid`, `accepted_ts` | broker 接受请求后的确认 |
| `Fill` | `broker_oid`, `symbol`, `qty`, `price`, `commission`, `ts` | 成交事件 |
| `PositionSnapshot` | `symbol`, `qty`, `avg_price`, `ts` | 持仓快照 |
| `AccountSnapshot` | `cash`, `equity`, `ts` | 账户快照 |
| `OrderStatusUpdate` | `broker_oid`, `status_str`, `ts`, `replay` | 订单状态更新 |

这些类型只描述跨运行时共同语义，不携带 Backtrader bracket 字段，也不携带某个券商私有状态机。

实盘适配器应同时支持单笔 `order_status(...)` 和批量 `order_statuses(...)`。执行器需要等待委托反馈时，优先批量查询，减少来源查询或 REST proxy 的请求次数。

## BrokerEventPump

内部事件泵用来把外部 broker 的轮询结果标准化成框架事件。普通用户不需要直接使用它；私有适配器作者如需对接 broker 事件，应参考内部契约文档，而不是在策略代码里导入 `tradelearn.core`。

## 策略侧约束

为了让同一套策略可以从回测迁移到 paper / live，建议遵守：

1. 在 `next()` 里表达交易意图：`buy`、`sell`、`close`、`order_target_percent`、`target_weights`。
2. 不假设立即成交，也不要假设发单后持仓立即变化。
3. 通过 `notify_order`、`notify_trade`、Stats 或 broker 事件观察成交结果。
4. 多资产目标权重策略应表达“目标组合”，不要在策略里手写 broker 状态同步。

## 外部适配器与 Nautilus 依赖

券商与执行宿主适配器应作为独立扩展实现：

- 输入：`OrderRequest` / 策略意图。
- 输出：`Fill` / `OrderStatusUpdate` / `PositionSnapshot` / `AccountSnapshot`。
- 框架内：只依赖 `tradelearn.core` 契约和事件泵。

这样可以保持 `tradelearn.backtest` 的 Rust 性能优化和实盘 broker 的连接逻辑互不污染。

[NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 是公开的外部交易引擎。选择它作为执行宿主时，集成层必须安装并导入 `nautilus_trader` 及所需适配器；业务策略可以继续使用 TL API。公共 TL 本身不强制依赖 Nautilus。

0.2.9将通用桥接和可选日志纳入同一发行包；安装 `trade-learn[nautilus]==0.2.9` 才装配固定官方框架，桥接不强制启用商业监控。基础包不导入Na，未装依赖或Python版本不支持时，实际桥接入口明确拒绝。本地自研适配器不随公共包交付，须独立核对是否存在于目标上游版本。接入方应声明并验证支持的版本、平台、订单能力、恢复语义和报告顺序。

## 当前源码状态

| 能力 | 状态与限制 |
|---|---|
| 原生回测、Engine / Lite 研究 | 已有实现；原生回测需匹配的 Rust 扩展 |
| 正式 StrategyRuntime | 0.2.9 覆盖默认研究与增量事件协调；不启动内部 loop |
| Broker 中性契约、内部事件泵 | 接入基础设施；内部类型不是策略 facade |
| 通用只读 observers | capability 1；不含持久传输或商业 SDK |
| Nautilus通用桥 | 包内 `tradelearn_runtime.execution`；可选官方rc7，账户/路由需显式配置，具体venue另验 |
| 具体券商接入 | 官方内置或独立扩展；不交付私有cTrader、账户配置或生产策略 |
| 发行与实盘验收 | 单独核验；源码存在不表示已发布或已通过真实券商验收 |
