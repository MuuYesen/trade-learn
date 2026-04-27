# 组合记账

## 范围

本文记录未来 Rust 回测内核的 clean-room portfolio 记账模型。范围包括现金、持仓、权益、已实现 PnL、未实现 PnL、手续费、滑点、做空持仓和多资产聚合。

本文关注内部账户状态以及通过 backtrader 风格 broker 门面对外暴露的值。报告生成和 factor 分析只消费已完成的 portfolio 输出,不在本文中定义。

本文不定义订单是否成交;撮合结果通过 fill 事件进入 portfolio。撮合细节由 `matching-design.md` 约束。

## Clean-room 边界

记账实现必须从项目 specs 和本文推导。不得复制外部回测引擎的 portfolio、broker 或 ledger 代码。

外部系统只用于理解常见会计术语与职责划分。下面的公式和对象边界都以 trade-learn 规则重新表述,后续实现必须独立完成。

公开 Python API 需要贴近 backtrader 用户习惯,但内部 Rust ledger 不需要暴露 backtrader 的对象结构。

## 来源笔记

来源清单:

- [x] `docs/specs/BACKTEST_SPEC.md`:portfolio accounting、保证金、commission/slippage 边界
- [x] `docs/specs/CONTRACTS.md`:Broker contract
- [x] `docs/specs/REPORT_SPEC.md`:下游 report artifacts
- [x] kernc/backtesting.py `backtesting/backtesting.py`:`Position`、`Trade`、`_Broker.equity`,仅作概念阅读
- [x] mementum/backtrader `backtrader/position.py` 和 `backtrader/brokers/bbroker.py`:`Position.update`、`BackBroker._execute`,仅作概念阅读
- [x] nautechsystems/nautilus_trader `crates/portfolio/src/portfolio.rs`:unrealized/realized PnL 和 update 入口,仅作概念阅读
- [x] vnpy/vnpy_ctastrategy `vnpy_ctastrategy/backtesting.py`:`calculate_result` / `DailyResult`,仅作概念阅读

从项目规格抽出的决策:

- Equity 等于 cash 加上持仓市值。
- 未实现 PnL 基于当前 mark price 与平均入场价。
- 已实现 PnL 累积已平仓结果。
- 手续费在成交时立即从 cash 扣除。
- 滑点通过成交价体现,不单独再扣费用。
- 每个 symbol 独立记录 position,账户 equity 做全局聚合。

外部源码概念观察:

- backtesting.py 通过 broker、position、trade 三类概念暴露账户状态。trade-learn 应保持公开 Broker facade 接近 backtrader 用户习惯,但真实 ledger 状态由 Rust core 持有并输出稳定 artifacts。
- backtesting.py 将权益计算放在 broker 状态上,并把成交价、手续费、持仓变化连接在一起。trade-learn 需要把这些动作拆成可测试事件:fill 先改变现金和 position,bar mark 再刷新 unrealized PnL 和 equity。
- backtrader 的 `Position.update` 负责根据成交 size/price 推导 opened / closed / adjusted 状态,`BackBroker._execute` 再处理现金、佣金和持仓更新。trade-learn 可参考这个职责分层:position 只处理数量与均价,broker/portfolio 处理现金、费用和账户聚合。
- backtrader 用户通常通过 `broker.getcash()`、`broker.getvalue()`、`position` 等接口观察账户。trade-learn Python 门面需要对齐这些可见行为,但不要暴露 Rust 内部 ledger 结构。
- NautilusTrader 将 portfolio update 与 matching 分离,并区分 account、order、position、quote、bar 等输入路径。trade-learn 也应避免在 matcher 内隐式改 PnL,而是让成交事件和行情标记事件分别驱动 portfolio。
- vn.py CTA 用逐日 `DailyResult` 汇总交易盈亏和持仓盈亏。trade-learn report artifacts 应同时支持 trade-level 与 daily-level 复算,这样 Stage 3 金标能比较 trades、PnL 和 drawdown 三类结果。

## 实现决策

portfolio 状态拆成账户级 cash 与每个 symbol 的 position。position 记录 signed size、average price、realized PnL、unrealized PnL 和 latest mark price。

每次 fill:

- 先通过成交价体现滑点。
- 立即从 cash 扣除 commission。
- 根据 signed fill direction 增加、减少、关闭或反转 symbol position。
- 只对减少既有 position 的数量确认 realized PnL。
- 对剩余或新开的 position 重新计算 average price。

每次 bar mark:

- 用可用行情更新每个 symbol 的 mark price。
- 重新计算 unrealized PnL。
- 将 account equity 计算为 cash 加 position market value。

做空持仓用负 size 表示。第一版采用 1:1 保证金处理,不计算融券利息,与 Stage 2 规格边界一致。

报告输出至少需要稳定生成 trades、orders、fills、positions、daily equity 和 drawdown 所需字段,以便 Stage 3 golden tests 可以分别比较交易层和账户层。

## 开放问题

单笔 fill 反转持仓的 1.0 行为已冻结为:订单与 fill 仍保持一笔成交,position ledger 在同一成交价上先关闭旧方向再打开剩余新方向;trade artifacts 与 `notify_trade()` 显式拆成两段,先输出 close leg(`size=0`,`isclosed=True`,记录旧仓位已实现 PnL),再输出 open leg(`size=反转后的新仓位`,`isopen=True`,`pnl=0`)。若该 fill 有 commission,按 close/open 的绝对成交数量比例分摊到两段 trade;order/fill 记录仍保留整笔 commission。

report artifacts 需要在 portfolio ledger 冻结前确定 positions、orders、fills、trade records 的稳定 schema。

现金不足和保证金不足的错误路径已冻结为 `Rejected` 订单状态。买入订单在成交前检查
`cash >= notional + commission`;开空订单按 1:1 保证金检查新增空头名义金额加
commission,不足时直接拒绝,不允许负现金,也不做 partial fill。平多产生的卖出成交只需
覆盖 commission,不按开空保证金处理。若未来版本需要 partial fill、融资融券或允许负现金,
必须先扩展订单状态、portfolio ledger 与 Analyzer 通知语义。
