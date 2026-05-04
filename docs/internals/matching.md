# 撮合与成交

本页定义 trade-learn Rust 撮合核在 bar 边界上的成交规则——每种订单类型如何触发、用什么价格成交、滑点和手续费如何处理。

> 撮合规则是回测**可复现性**的来源。理解它，你就能解释每一笔成交的价格、能预知 `next()` 中下的单何时被吃掉，也能在和外部引擎对照时定位差异。

## 订单类型与触发条件

trade-learn 2.0 支持四种订单类型：

| 类型 | 何时触发 | 成交价 |
|---|---|---|
| **Market（市价）** | 立即（下一根 bar 的执行时点） | 配置的执行时点价（默认 next open）+ 滑点 |
| **Limit（限价）** | 下一根 bar 的 high/low 触达 limit 时 | 买：`min(limit, next_open)`<br>卖：`max(limit, next_open)` |
| **Stop（止损）** | high/low 触达 stop 时 | 转为市价语义 |
| **Stop-Limit（止损限价）** | stop 触发后再按 limit 规则撮合 | Python 门面：`price` 表示 stop 触发价、`pricelimit` 表示 limit 成交价 |

## 默认成交时点

- 市价单**默认在下一根 bar 的 open 成交**——`next()` 中下的单进入下一次撮合机会
- `trade_on_close=True` 时，当前 `next()` 创建的市价单允许在**当前 bar close** 成交

详见 [设计笔记 → 事件循环](event-loop.md)。

## bar 内 stop + limit 同时触达

在同一根 OHLC bar 内不推断真实 tick 路径。若 stop 与 limit 在同一根 bar 内均可见，按确定性规则处理：

1. 先用 high / low 判定 stop 是否被触发
2. 再用同一根 bar 的 high / low 判定 limit 是否可成交
3. 买入限价成交价为 `min(limit_price, open)`，卖出为 `max(limit_price, open)`

这条规则保证回测可重放、不依赖真实 tick 序。

## 订单生命周期

订单状态机显式保留以下状态：

```
submitted → accepted → filled
                     ↘ partially_filled  (2.0 不支持，预留)
                     ↘ cancelled
                     ↘ rejected
                     ↘ expired
```

`time_in_force` 控制过期：

| 取值 | 语义 |
|---|---|
| `day` | 当日有效 |
| `gtc` | good-till-cancelled |
| `ioc` | immediate-or-cancel |

## 2.0 不支持 Partial Fill

若订单数量超过当前 bar 可用 `volume`，订单**直接 rejected**，不会按 volume 部分成交。

> 这是有意为之：partial fill 会污染 Analyzer 的盈亏统计与 trade 闭环判定。后续版本若要支持，需先扩展订单状态、fill 聚合和 `notify_trade` 语义。

## 现金 / 保证金不足

下单前检查：

- **买入订单**：`cash ≥ notional + commission`，否则 `Rejected`
- **开空订单**：按 1:1 保证金检查"新增空头名义金额 + commission"，不足则 `Rejected`
- **平多产生的卖出**：只需覆盖 commission

trade-learn 2.0 **不允许负现金、不做 partial fill**。

## 反转持仓的 trade 拆分

单笔 fill 反转持仓（如持多 100 股，下卖出 200 股）时：

- 订单和 fill **仍记录为一笔成交**
- Position ledger 在同一成交价上**先关闭旧方向再打开剩余新方向**
- `trade artifacts` 与 `notify_trade()` 显式拆成两段：
  1. **close leg**：`size=0`、`isclosed=True`、记录旧仓位的已实现 PnL
  2. **open leg**：`size=反转后的新仓位`、`isopen=True`、`pnl=0`
- 若该 fill 有 commission，按 close / open 的绝对成交数量比例分摊到两段 trade
- 原始 order / fill 记录**保留整笔 commission**

## 精度规则（已冻结为 6 位小数）

| 量 | 规则 |
|---|---|
| 订单 size | 不做 lot-size 转换，进入 fill 事件前统一四舍五入到 6 位小数 |
| 成交价 | 先按订单类型得到 raw price → 按方向应用滑点 → 四舍五入到 6 位小数 |
| 百分比手续费 | 按精度处理后的成交价与 size 计算，再四舍五入到 6 位小数 |
| 固定手续费 | 进入同一现金精度（6 位小数） |
| `FillEvent.slippage` | 精度处理后的成交价与 raw price 之差，四舍五入到 6 位小数 |

> 2.0 **不**做 A 股 100 股手数自动取整。需要的话用户在策略层显式 round。

## 与 backtrader 的语义对齐

trade-learn 撮合规则的设计目标：

- 用户 API（`buy` / `sell` / `close` / `notify_order` / `notify_trade`）保持 backtrader 风格
- 撮合层语义（成交时点、limit / stop 触发条件、partial fill 不支持）走 trade-learn 自己的 spec
- `tests/golden/` 用 backtrader 作为 oracle，对兼容策略要求 **trades 0 差异**、equity `rtol=1e-6`、summary `rtol=1e-4`

详见 [与 backtrader 的语义一致性](consistency.md)。

## 相关阅读

- [事件循环](event-loop.md)：撮合在每根 bar 中的位置
- [Portfolio 模型](portfolio.md)：fill 进入 portfolio 后如何变成持仓与现金
- [契约与边界](contracts.md)：Order / Fill / OrderStatus 的字段定义
