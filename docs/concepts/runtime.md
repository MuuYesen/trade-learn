# Runtime 与 Runner

trade-learn 是事件驱动回测框架，不是纯向量化回测器。策略仍然在每根 bar 上通过 `next()` 表达逻辑，runtime 负责把当前 bar、订单、成交、持仓和权益状态推进下去。

## 自动 runner 选择

| 场景 | 自动路径 | 触发条件 |
|---|---|---|
| 单标的 | Rust single-data runner | `RustBroker` 已绑定 Rust engine，且只有 1 个 data feed |
| 多标的 | Rust multi-data clock runner | `len(datas) > 1`，每个 feed 暴露 OHLCV 数组 |
| 自定义 feed / 非 Rust broker | Python fallback | 不满足数组协议或 broker 条件 |

用户不需要显式选择 runner。Engine 和 Lite 都通过 `tradelearn.backtest` runtime 进入同一套内核。

## 事件顺序

```mermaid
sequenceDiagram
    participant S as Strategy.next()
    participant BT as Python runtime
    participant RR as Rust runner
    participant RB as RustBroker
    participant ST as Stats

    BT->>RR: 推进 bar clock
    RR-->>BT: 当前 cursor / active bars
    BT->>S: 调用策略 next()
    S->>BT: 产生订单意图
    BT->>RB: 提交订单
    RB->>RB: 撮合、成交、更新 portfolio
    RB-->>BT: compact fills / state
    BT->>ST: 更新 summary / artifacts
```

## 为什么不是纯向量化

纯向量化可以很快，但会弱化订单生命周期、成交事件、拒单、撤单、现金和持仓状态回流。trade-learn 保留事件驱动模型，是为了让回测策略更容易迁移到 paper/live broker，同时把真正高频的撮合和状态推进放进 Rust。
