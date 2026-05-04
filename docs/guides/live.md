# 实盘协议

trade-learn 1.0 的主能力是事件驱动回测；paper / live 的边界已经通过 `tradelearn.core` 预留。理解这页可以帮助你写不会被回测语义绑死的策略和 broker 适配器。

## 回测 broker 与实盘 broker 的区别

| 项目 | Rust 回测 broker | paper / live broker |
|---|---|---|
| 职责 | 本地撮合、订单推进、portfolio 记账 | 连接外部交易系统，回传订单状态和成交 |
| 状态来源 | Rust 内核内部状态 | 外部 broker / 交易柜台 |
| 成交语义 | 按 bar 和撮合模式确定 | 由真实市场和 broker 回报决定 |
| 策略假设 | `next()` 内发单，后续事件更新状态 | 同样发单，但不能假设立即成交 |
| 接口边界 | `tradelearn.backtest.RustBroker` | `tradelearn.core.Broker` + `BrokerEventPump` |

因此，为了加快回测而优化 RustBroker，不会自动改变实盘 broker；但策略层必须只依赖共同的“意图 + 事件”语义。

## 中性订单协议

实盘适配器应该接收 `OrderRequest`，返回 `OrderAck`，并通过事件回流 `Fill`、撤单、拒单和状态。

| 类型 | 关键字段 | 用途 |
|---|---|---|
| `OrderRequest` | `symbol`, `side`, `qty`, `order_type`, `limit_price`, `stop_price`, `tif`, `client_oid` | 策略发出的 broker-neutral 下单请求 |
| `OrderAck` | `client_oid`, `broker_oid`, `accepted_ts` | broker 接受请求后的确认 |
| `Fill` | `broker_oid`, `symbol`, `qty`, `price`, `commission`, `ts` | 成交事件 |
| `PositionSnapshot` | `symbol`, `qty`, `avg_price`, `ts` | 持仓快照 |
| `AccountSnapshot` | `cash`, `equity`, `ts` | 账户快照 |
| `OrderStatusUpdate` | `broker_oid`, `status_str`, `ts`, `replay` | 订单状态更新 |

这些类型只描述跨运行时共同语义，不携带 Backtrader bracket 字段，也不携带某个券商私有状态机。

## BrokerEventPump

`BrokerEventPump` 用来把外部 broker 的轮询结果标准化成框架事件。

```python
from tradelearn.core import BrokerEvent, BrokerEventPump


def poller():
    # 从 QMT / IB / CTP / REST proxy 等外部系统拉取事件。
    return [
        BrokerEvent(kind="fill", order_id="O-1", payload=...),
        BrokerEvent(kind="status", order_id="O-2", status="accepted"),
    ]


pump = BrokerEventPump(poller)
pump.on_fill(lambda fill: print("fill", fill))
pump.on_status(lambda oid, status, replay: print(oid, status))
pump.poll_once()
```

## 策略侧约束

为了让同一套策略可以从回测迁移到 paper / live，建议遵守：

1. 在 `next()` 里表达交易意图：`buy`、`sell`、`close`、`order_target_percent`、`target_weights`。
2. 不假设立即成交，也不要假设发单后持仓立即变化。
3. 通过 `notify_order`、`notify_trade`、Stats 或 broker 事件观察成交结果。
4. 多资产目标权重策略应表达“目标组合”，不要在策略里手写 broker 状态同步。

## QMT / 私有适配器

QMT、IB、CTP、Binance 等适配器应作为独立扩展实现：

- 输入：`OrderRequest` / 策略意图。
- 输出：`Fill` / `OrderStatusUpdate` / `PositionSnapshot` / `AccountSnapshot`。
- 框架内：只依赖 `tradelearn.core` 契约和事件泵。

这样可以保持 `tradelearn.backtest` 的 Rust 性能优化和实盘 broker 的连接逻辑互不污染。

## 当前状态

| 能力 | 1.0 状态 |
|---|---|
| Rust 事件驱动回测 | 稳定 |
| Engine / Lite 策略入口 | 稳定 |
| Broker 中性契约 | 稳定 |
| `BrokerEventPump` | 可用 |
| 具体 QMT / live broker 包 | 建议外部扩展或私有部署 |
