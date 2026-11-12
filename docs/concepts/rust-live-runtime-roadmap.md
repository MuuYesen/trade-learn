# Rust Live Runtime 路线图

> **2026-11-11 修订：旧自建实盘运行时路线已由后续边界覆盖。** 当前方向是保持Python用户API、优化TL自身的原生回测/指标/数据运行层，实盘仍沿官方Nautilus。这里不新增执行框架或已实现能力。

现行职责、迁移顺序与完成条件唯一见[框架规范：Python接口与原生运行时分工](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#python-native-runtime-20261111)。指标后端另见[来源与流式资格](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#indicator-sources-streaming-20261111)。本页只映射当前源码，不复制规则。

## 当前源码分工

| 层 | 实现与入口 | 当前状态 |
|---|---|---|
| Python用户接口 | `tradelearn/engine/strategy.py`、`engine/cerebro.py`及Lite | 用户策略和研究编排继续为Python；Engine/Lite统一迁移另按主规范 |
| 策略协调 | `tradelearn/backtest/event_runtime.py` | Python StrategyRuntime管理生命周期、历史、暖机、指标和回调 |
| 回测编排 | `tradelearn/backtest/engine.py` | Python选择Runner、预载数据、收集结果；满足条件时进入Rust循环 |
| 回测内核 | `rust/tradelearn-rust/src/lib.rs`及engine/matching/runner模块 | RustBacktestEngine、撮合、订单/组合状态与部分时钟/Bar循环；循环可回调Python |
| 指标与因子 | `tradelearn/indicators/`、`factor/alpha/` | 目前主要以Python包装批量后端；新流式后端仍待逐项接入与验收 |
| 实盘执行 | 官方Nautilus、通用桥、独立适配器 | 不由RustBroker模拟撮合替代权威券商事实；具体来源能力单独验收 |

Python/Rust采用PyO3/maturin绑定，不等于所有对象零拷贝或整个事件路径无Python。原生编译后端可以是Rust、C或C++，用户接口和数值语义不按实现语言变化。

## 后续工作状态

共同生命周期/有界缓冲、合格流式指标、减少转换复制与跨语言往返、按证据下沉数据/调度工作均是后续目标；当前文档整理没有实现这些迁移。正式执行前固定源码/依赖/平台、参考结果和成本证据；不能用旧性能样本证明新路径。

回测RustBroker不扩为实盘网关，不新增TL常驻实盘循环；不要求Python策略改写成Rust，也不增加Tick或替换用户next生命周期。

## 旧路线的处理

旧稿中的`Engine(runtime="rust")`、`add_strategy`示例、Rust Strategy trait、TL自建QMT/CTP/FIX网关及双语言策略生态不是本轮承诺或现行API，已从可执行指南中移除。原来“Python接口可以调用原生底层”和“回测撮合不能冒充实盘执行”的原则保留；其余以主规范为准。

## 相关阅读

- [当前架构与边界](architecture.md)
- [实际Runtime契约](runtime.md)
- [当前实盘接入指南](../guides/live.md)
