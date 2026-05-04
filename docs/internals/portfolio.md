# Portfolio 模型

本页解释成交事件（`Fill`）如何变成现金、持仓和权益曲线——也就是 trade-learn 内部的**记账模型**。撮合规则定义"何时成交、成交价是多少"，portfolio 模型定义"成交后账户长什么样"。

## 状态拆分

portfolio 状态拆成两层：

```
Account
├── cash                      # 全局现金
└── positions: dict[symbol, Position]
    ├── size                   # signed（多头正、空头负）
    ├── average_price          # 加权入场价
    ├── realized_pnl           # 已实现盈亏
    ├── unrealized_pnl         # 未实现盈亏
    └── latest_mark_price      # 最近一次 mark 的价格
```

**Position 只记数量与均价；现金、手续费、账户聚合由 Account 处理。** 这条职责分层让单测变得简单：position 只对 fill 反应，account 只对 mark 反应。

## 每次 Fill 的处理流程

收到一条 `Fill` 事件时，引擎按以下顺序更新账户：

1. **滑点**通过成交价直接体现（不再单独扣费）
2. **从 cash 立即扣除 commission**
3. 根据 signed fill direction（buy 为正、sell 为负）**更新 symbol position**：
   - 增加既有方向 → 重新计算 average price
   - 减少既有方向 → 确认这部分的 realized PnL（average price 不变）
   - 关闭 → 全量 realized
   - 反转 → 先 close leg 再 open leg（详见 [撮合 → 反转持仓的 trade 拆分](matching.md#trade)）

## 每次 Bar Mark 的处理流程

每根 bar 推进时（在 `strategy.next()` 之前），引擎做 mark-to-market：

1. 用当前 bar close 更新每个 symbol 的 `latest_mark_price`
2. 重新计算 `unrealized_pnl = (mark - avg_price) * size`
3. 更新账户 equity：

```
equity = cash + Σ (mark_price * size) for all positions
```

## 做空持仓

用**负 size** 表示。2.0 采用 1:1 保证金处理，不计算融券利息——这是当前回测边界，与现实差距已知，但保证回测复现性。

## 报告口径

portfolio 输出至少稳定生成以下字段，作为 [Stats](../concepts/stats.md) 的来源：

| 表 | 内容 |
|---|---|
| `trades` | 逐笔交易（开 + 平闭环） |
| `orders` | 订单历史（含 cancelled / rejected） |
| `fills` | 逐笔成交 |
| `positions` | 每日持仓快照 |
| `equity` | 每日权益曲线 |
| `drawdown` | 每日回撤 |

> 核心对齐测试 (Golden Test) 同时比较 trade-level 和 daily-level 结果，因此两个层面的 schema 都必须稳定。

## 公开 API（backtrader 兼容）

用户在策略中通过这些方法观察账户：

```python
self.broker.getcash()       # 当前现金
self.broker.getvalue()      # 当前权益
self.position.size          # 当前持仓数量
self.position.price         # 当前持仓均价
```

这些方法的行为和 backtrader 一致——用户**不直接接触 Rust 内部 ledger**。

## 已冻结的边界行为

以下行为在 2.0 已冻结，更改需走 breaking change 流程：

- **现金不足 / 保证金不足 → `Rejected`**，不允许负现金，不做 partial fill
- **单笔反转 fill** → 拆成 close + open 两段 trade（commission 按比例分摊）
- **做空** → 1:1 保证金，不计算融券利息

后续版本若要支持 partial fill、融资融券或允许负现金，必须先扩展订单状态、portfolio ledger 与 Analyzer 通知语义。

## 相关阅读

- [事件循环](event-loop.md)：mark-to-market 在每根 bar 中的位置
- [撮合与成交](matching.md)：Fill 事件来源
- [Stats 结果对象](../concepts/stats.md)：portfolio 输出落地为用户可访问的字段
