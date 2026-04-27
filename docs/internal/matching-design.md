# 撮合设计

## 范围

本文记录未来 Rust 回测内核的 clean-room 撮合规则。范围包括市价单、限价单、止损单、止损限价单在 bar 边界上的处理方式,以及订单提交时点与 `trade_on_close` 的关系。

设计目标是 Stage 3 的撮合模块。Python 策略 API 仍属于 backtrader 风格门面层;本文只定义 Rust core 需要独立实现的撮合语义。

本文不定义 portfolio 记账细节、analyzer 行为或报告格式。这些内容分别由 `portfolio.md`、`event-loop.md` 与 REPORT_SPEC 约束。

## Clean-room 边界

实现必须以项目 specs 与本文为工作来源。不得把 backtesting.py、backtrader、NautilusTrader、vn.py 或其他外部引擎的实现代码、测试、注释、私有算法复制进 Rust core。

外部项目只用于概念层面对照,例如事件驱动撮合、订单生命周期、bar 内成交时点、滑点和手续费抽象。本文所有最终规则都用 trade-learn 自己的术语重新表述,后续实现必须独立编写。

阅读外部源码时只记录行为观察和设计启发,不记录可直接搬运的代码片段。Stage 3 实现时仍以本文和 BACKTEST_SPEC 为准。

## 来源笔记

来源清单:

- [x] `docs/specs/BACKTEST_SPEC.md`:撮合规则、订单类型、`trade_on_close` 语义
- [x] `docs/specs/STRATEGY_SPEC.md`:用户侧下单 API 与 backtrader 风格约束
- [x] `docs/specs/CONSISTENCY.md`:决策层零差异目标
- [x] kernc/backtesting.py `backtesting/backtesting.py`:`_Broker._process_orders`、`trade_on_close`、market/limit/stop/stop-limit 分支,仅作概念阅读
- [x] mementum/backtrader `backtrader/brokers/bbroker.py`:`BackBroker._try_exec*`、`BackBroker.next`、订单状态推进,仅作概念阅读
- [x] nautechsystems/nautilus_trader `crates/execution/src/matching_engine/engine.rs`:`OrderMatchingEngine` 的订单类型分派、触发与 fill 入口,仅作概念阅读
- [x] vnpy/vnpy_ctastrategy `vnpy_ctastrategy/backtesting.py`:`cross_limit_order` / `cross_stop_order`,仅作概念阅读

从项目规格抽出的决策:

- 市价单默认在下一根 bar 的 open 成交。
- `trade_on_close=True` 时,`next()` 中创建的订单允许在当前 bar close 成交。
- 买入限价单在下一根 bar 的 low 触达 limit 时成交;卖出限价单在下一根 bar 的 high 触达 limit 时成交。
- 止损单由 high 或 low 触发,触发后转为市价语义。
- 止损限价单先由 stop 条件触发,再按 limit 规则撮合。
- Stage 3 golden fixture 建立后,决策层目标是 trades 零差异。

外部源码概念观察:

- backtesting.py 将基于 bar 的撮合集中放在 broker 内部,`trade_on_close` 是成交时点开关。trade-learn 应保留规格里"默认下一根 open、可选当前 close"的语义,但最终 Rust 实现不照搬其 Python 控制流。
- backtesting.py 对市价、限价、止损、止损限价使用同一个订单处理入口再按类型分支。trade-learn 可以借鉴"统一入口 + 类型分派"这个结构思想,但订单状态、价格规则和错误语义以 BACKTEST_SPEC 为准。
- backtrader 的 `BackBroker._try_exec_market/_limit/_stop/_stoplimit` 明确把不同订单类型拆成独立尝试成交路径,并通过 broker `next` 推进未完成订单。trade-learn 的用户 API 对齐 backtrader,因此 Python 门面应保持订单对象和状态回调风格相近;Rust 内核则只暴露稳定事件与成交结果。
- NautilusTrader 的 Rust matching engine 将订单类型处理、触发、成交写成明确入口,并保留更丰富的订单类型扩展位。trade-learn 1.0 只实现四类订单,但应预留清晰枚举和状态机,避免未来加入 trailing/OCO 时重写主循环。
- vn.py CTA 在 `new_bar` 中先做限价/停止单穿越判断,再调用策略 `on_bar`。这支持当前设计:上一根遗留订单必须先处理,再允许用户策略基于当前 bar 产生新订单。

## 实现决策

撮合器应在当前 bar 的 Python `next()` 运行前处理待撮合订单。`next()` 产生的新订单进入下一次撮合机会,除非 `trade_on_close=True`。

确定性的 bar 级成交规则:

- 市价买单和卖单使用配置的执行时点价格,再应用滑点。
- 买入限价单触达条件满足后,成交价为 `min(limit_price, next_open)`。
- 卖出限价单触达条件满足后,成交价为 `max(limit_price, next_open)`。
- 止损市价单只用 stop 触发条件决定是否激活,最终成交价遵循市价单规则。
- 止损限价单触发后必须在订单状态中同时保留 stop 触发信息和 limit 价格;Python 门面使用 `price` 表示 stop 触发价,使用 `pricelimit` 表示 limit 成交价。
- 同一根 OHLC bar 内不推断真实 tick 路径。若 stop 与 limit 在同一根 bar 内均可见,按确定性规则处理:先判定 stop 是否被 high/low 触发,再用同一根 bar 的 high/low 判定 limit 是否可成交;买入限价成交价为 `min(limit_price, open)`,卖出限价成交价为 `max(limit_price, open)`。

第一版实现应显式保留订单生命周期:submitted、accepted、filled、cancelled、rejected、expired。过期逻辑由 `time_in_force` 驱动,首批支持 day、good-till-cancelled、immediate-or-cancel。

Rust 内核应输出 Fill/Cancel/Reject 事件,由 Python 层再转成 backtrader 风格的 `notify_order` / `notify_trade` 回调。这样可以同时满足 clean-room 实现和用户 API 对齐。

1.0 不支持 partial fill。若订单数量超过当前 bar 可用 volume,不会按 volume 部分成交,而是直接 rejected;后续版本如需支持 partial fill,必须先扩展订单状态、fill 聚合和 Analyzer 通知语义。

撮合精度规则冻结为:

- 订单 size 不做 lot-size 转换,但进入 fill 事件前统一四舍五入到 6 位小数;1.0 不隐式按 A 股 100 股手数取整。
- 成交价先按订单类型得到 raw price,再按买卖方向应用滑点,最后统一四舍五入到 6 位小数。
- 百分比手续费按精度处理后的成交价与 size 计算,再四舍五入到 6 位小数;固定手续费也进入同一现金精度。
- `FillEvent.slippage` 记录精度处理后的成交价与 raw price 的差值,同样四舍五入到 6 位小数。

## 开放问题

Stage 3 matching 精度规则已冻结为 6 位小数。后续若要支持交易所 lot-size、tick-size 或最小佣金,必须先扩展订单参数、broker 配置和 golden fixture。
