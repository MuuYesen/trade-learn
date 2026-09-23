# 撮合与成交

本页定义 trade-learn Rust 撮合核在 bar 边界上的成交规则——每种订单类型如何触发、用什么价格成交、滑点和手续费如何处理。

> 撮合规则是回测**可复现性**的来源。理解它，你就能解释每一笔成交的价格、能预知 `next()` 中下的单何时被吃掉，也能在和外部引擎对照时定位差异。

## 订单类型与触发条件

trade-learn 0.2.6 支持以下订单类型（表内价格为应用滑点前的 raw price）：

| 类型 | 何时触发 | 成交价 |
|---|---|---|
| **Market（市价）** | 立即（下一根 bar 的执行时点） | 配置的执行时点价（默认 next open）+ 滑点 |
| **Limit（限价）** | 下一根 bar 的 high/low 触达 limit 时 | 买：`min(limit, next_open)`<br>卖：`max(limit, next_open)` |
| **Stop（止损）** | high/low 触达 stop 时 | 默认 next-open 模式：买入跳空高开取 open、盘中触达取 stop，即 `max(open, stop)`；卖出反向，即 `min(open, stop)` |
| **Stop-Limit（止损限价）** | stop 触发后再按 limit 规则撮合 | Python 门面：`price` 表示 stop 触发价、`pricelimit` 表示 limit 成交价 |
| **StopTrail（跟踪止损）** | 触达上一根有效 bar 保存的跟踪线 | 按 Stop 的跳空 / 触达规则成交 |
| **StopTrailLimit（跟踪止损限价）** | 跟踪线触发后成为持续有效的限价单 | 仅在触发之后满足 `pricelimit` 时成交 |

## 默认成交时点

- 市价单**默认在下一根 bar 的 open 成交**——`next()` 中下的单进入下一次撮合机会
- `trade_on_close=True` 时，当前 `next()` 创建的市价单允许在**当前 bar close** 成交

详见 [设计笔记 → 事件循环](event-loop.md)。

## bar 内 stop + limit 同时触达

普通 `StopLimit` 在 exact 模式下沿用以下规则；OHLC 无法还原真实 tick 顺序：

1. 先用 high / low 判定 stop 是否被触发
2. 再用同一根 bar 的 high / low 判定 limit 是否可成交
3. 买入限价成交价为 `min(limit_price, open)`，卖出为 `max(limit_price, open)`

smart 模式采用确定性路径：阳线 / 平线为 `open → low → high → close`，阴线为 `open → high → low → close`。这是一项可复现的撮合约定，不代表真实 tick 路径。

`StopTrailLimit` 在 exact 与 smart 模式都使用上述路径，且只检查触发点之后的限价机会；触发之前触达限价不能倒推成交。跳空越过跟踪线但未满足限价时，订单保留已触发状态，之后的 bar 只按限价撮合。

## 跟踪水位

`trailamount` 指定绝对回撤，`trailpercent` 指定比例回撤（例如 `0.05`）；同时指定时优先使用 `trailamount`。`price` 是初始参考水位，Python 门面省略时以当前 `close` 初始化。卖出水位只升不降，买入水位只降不升，初始参考价也参与比较。

每根 bar 先按已保存水位撮合，未成交订单再用本 bar 极值推进下一根 bar 的水位，避免用当前 high 抬高水位后回看当前 low。无成交量的 bar 不推进水位。跟踪限价单一旦触发便冻结水位。

## 订单生命周期

订单沿用现有 Python broker 与 Rust matcher 的职责划分：broker 创建、通知并保存订单元数据，Rust 在撮合前处理有效期，并在成交时原子撤销显式 OCO 同组订单。终态回传 broker，由 `notify_order()` 通知策略并清理待成交数量。

```text
Created → Submitted → Accepted → Completed
                            ↘ Canceled / Expired / Margin / Rejected
```

- `valid=None` 表示无截止时间；`date` / `datetime` 表示绝对截止时刻（日期按当天零点），`timedelta` / 数字秒数相对提交时的主时钟计算一次。
- 无时区值按 UTC 解释，有时区值换算为 UTC。仅当当前时间严格晚于截止时刻才过期，截止时刻本身仍可成交。
- single-data Rust runner、multi-data Rust runner 和 Python fallback 均在撮合前执行过期检查；过期订单不会先成交再被标记失效。
- 显式 `oco=other_order` 在 exact 与 smart 模式都互斥；一笔成交即撤销仍存活的同组订单。smart 原有隐式保护性退出互斥规则为兼容性保留。
- 手动撤单、OCO 撤单和过期均同步原生待撮合队列与 Python 状态；关联策略收到相应的终态通知。括号订单的子单在父单成交后激活，父单终止时撤销尚未激活的子单。
- 原有 `Order` 数值常量保持兼容；`tag` 和 `info` 元数据可随订单 / 交易统计保留。

## 2.0 不支持 Partial Fill

当前撮合采用整单成交，不根据 bar 的 `volume` 拆分成交量；零成交量 bar 不成交。

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
