# 事件循环

## 范围

本文定义未来 Rust 回测内核与 Python callback 桥接层的 clean-room 事件循环顺序。范围包括 bar 输入、遗留订单撮合、mark-to-market、策略 callback 时点、analyzer 时点和暖机期行为。

本文只覆盖 Stage 3 回测引擎。数据归一化、factor 分析和报告生成是事件循环输出的消费者,不在本文中定义。

事件循环必须服务于 backtrader 风格用户 API:用户在 `Strategy.__init__()` 中声明指标,在 `next()` 中读取当前 bar 和指标值,通过 `buy` / `sell` / `close` 等 API 产生订单。

## Clean-room 边界

实现的事实来源是项目 specs 与本文。不得复制 backtesting.py、backtrader、vn.py、NautilusTrader 或其他交易引擎的 engine 代码、callback 机制、测试或注释。

外部参考只用于比较概念,例如事件队列、callback 生命周期、analyzer hook 和多数据源推进方式。最终 trade-learn 事件顺序必须独立表述,并能追溯到项目规格。

Stage 3 进入实现后,如果本文与外部源码行为不一致,以本文和 BACKTEST_SPEC 为准;需要改变时先更新本文和 PROGRESS,再实现。

## 来源笔记

来源清单:

- [x] `docs/specs/BACKTEST_SPEC.md`:事件循环顺序、暖机期、Python callback 时点
- [x] `docs/specs/ARCHITECTURE.md`:Python/Rust 边界与依赖方向
- [x] `docs/specs/STRATEGY_SPEC.md`:Strategy 生命周期、`next`、`notify_order`、`notify_trade`
- [x] kernc/backtesting.py `backtesting/backtesting.py`:`Backtest.run`、broker `next`、strategy 迭代关系,仅作概念阅读
- [x] mementum/backtrader `backtrader/cerebro.py`:`Cerebro.run`、`runstrategies`、`_runnext` / `_runonce`,仅作概念阅读
- [x] nautechsystems/nautilus_trader `nautilus_trader/backtest/engine.pyx` 和 `crates/backtest/src/engine.rs`:回测 engine 边界,仅作概念阅读
- [x] vnpy/vnpy_ctastrategy `vnpy_ctastrategy/backtesting.py`:`run_backtesting` 和 `new_bar`,仅作概念阅读

从项目规格抽出的决策:

- 每根新 bar 到达后,先更新引擎当前时间。
- 上一根 bar 遗留的订单在 `strategy.next()` 前撮合。
- 当前 bar close 用于 mark-to-market。
- Python `next()` 可以产生新订单。
- Analyzer 的 `on_bar()` 在当前 bar 的策略逻辑之后运行。
- Rust 负责撮合与记账;Python 负责策略逻辑与 callback。

外部源码概念观察:

- backtesting.py 将策略迭代与 broker 推进绑定在同一回测循环中。trade-learn 不复制该循环,但保留一个关键约束:用户策略看到当前 bar 前,引擎已经完成必要的订单处理与状态更新。
- backtesting.py 的 `trade_on_close` 影响订单何时进入撮合窗口。trade-learn 应把这个选择放在 `Cerebro` / engine 配置里,不要让 Strategy 自己绕过事件顺序直接成交。
- backtrader 的 `Cerebro` 同时支持逐 bar (`_runnext`) 与预计算 (`_runonce`) 路径。trade-learn 1.0 为保证 Rust/Python 边界清晰,应先固定逐 bar 事件语义;未来若做批量优化,也必须保持 `next` 可观察行为一致。
- backtrader 的策略调度围绕 datas、broker、analyzers 协同推进。trade-learn Python 门面应复刻用户可见的调用风格,但 Rust 内核只接收已归一化事件并返回 fills / portfolio marks,避免把 Python 对象生命周期塞进内核。
- NautilusTrader 将高层 backtest engine、execution、portfolio 拆成不同子系统。trade-learn Stage 3 也应拆成 engine loop、matching、portfolio 三个边界,后续 live adapter 才能复用 Broker/DataFeed trait。
- vn.py CTA 在 `new_bar` 里先做订单穿越,再调用策略 `on_bar`。这与本设计的"先处理遗留订单,再触发用户策略"一致,同时提醒我们 tick/bar 两类输入要共享相同的状态推进约束。

## 实现决策

事件循环使用确定性的逐 bar 顺序:

1. 读取下一根 bar,更新当前时间戳。
2. 撮合符合条件的遗留订单,产生 fill、cancel 或 reject 事件。
3. 使用当前 close 更新 portfolio mark。
4. 暖机期结束后调用 Python `strategy.next()`。
5. 将 Python callback 创建的订单入队。
6. 调用当前 bar 的 analyzers。
7. 推进到下一根 bar。

暖机期由已注册指标的 lookback 推导。最长 lookback 可用之前,循环可以更新内部数据缓冲和 portfolio mark,但不得调用用户 `next()`。

Python/Rust 边界应保持窄接口。Rust 可以批量处理数据和引擎事件;Python callback 收到稳定对象,并提供 backtrader 风格的当前 bar 索引语义。

多数据源场景下,事件队列必须保证确定性排序。相同时间戳的不同 data feed 需要固定 tie-break 规则,避免同一策略在不同平台上出现不同订单顺序。

Python `Cerebro` 门面采用主数据时钟推进策略 callback。`datas[0]` 是 primary
feed,每根 primary bar 都触发一次 broker / strategy / analyzer 生命周期;secondary
feed 在每个 primary timestamp 对齐到自身 `timestamp <= primary timestamp` 的最新
bar。若 secondary feed 在当前 primary timestamp 前还没有任何 bar,其 line cursor
保持未就绪状态,用户读取 `data.close[0]` 等 line 时得到与普通越界访问一致的
`IndexError`。该规则避免用最短数据长度截断主数据,也避免缺失 bar 时提前窥视未来
secondary bar。

## 已冻结语义

Analyzer 在同一 bar 内通过 `on_order(order)` 同步收到 submitted / accepted /
completed / cancelled / rejected / expired 订单状态变化。`on_fill()` 与
`on_trade()` 仍只处理成交相关事件,避免非成交订单污染成交统计。

多周期数据的 bar 对齐规则已冻结为 primary-clock + secondary latest-at-or-before
语义。后续若新增 Rust 批处理 callback 或 live adapter,必须保持 Python 用户可见的
`next()` 调用次数与 secondary line 可见性一致。

`callback_batch=1` 是默认逐 bar 语义,策略在当前 bar 创建的普通订单从下一根 bar
开始具备撮合资格;`trade_on_close=True` 且 `callback_batch=1` 时,当前 `next()` 创建的
market order 可在当前 bar close 撮合。`callback_batch=N>1` 时,策略在当前 bar
创建的订单延迟 N 根 primary bar 后才具备撮合资格;该规则让 Python 门面与
BACKTEST_SPEC 的批处理性能模式保持一致,也避免批处理路径中订单提前生效。

## 1.1 Rust 驱动事件循环

### 目标

将 bar 迭代循环从 Python `Cerebro.run()` 下沉至 Rust `BarRunner`，同时保持用户 API
100% 不变。`strategy.next()`、`self.data[0]`、`self.buy()` 等方法签名与语义均冻结，
策略作者无需改动任何代码。

### 动机

当前 1.0 架构中 Python for 循环每根 bar 承担调度开销，Rust 撮合核仅覆盖撮合与记账。
10000 bars × Python 调度开销约占总耗时 60%。将循环下沉 Rust 后，预计 vs 当前 ≥ 3x、
vs Backtrader ≥ 8x（sma_cross + 10000 bars 场景）。

### 架构变更

当前（1.0）：

```
Python Cerebro.run()
  └─ for bar in data:          # Python 驱动
       match_pending_orders()  # → Rust
       mark_to_market()        # → Rust
       strategy.next()         # Python 调 Python
```

目标（1.1）：

```
Python Cerebro.run()
  └─ rust_bar_runner.run(strategy_ptr)   # PyO3，一次性委托
       └─ Rust BarRunner（每根 bar）
            ├─ advance bar buffer         # Rust 写，Python property 读
            ├─ match_pending_orders()     # 已有
            ├─ mark_to_market()          # 已有
            ├─ PyO3 → strategy._pre_next()   # 推进 Python 指标
            └─ PyO3 → strategy.next()        # 用户逻辑不变
```

### 兼容性保证

- `strategy.next()`、`self.data[0]/[-1]`、`self.buy()/sell()/close()` 签名和语义不变。
- `self.fast[0]`（SMA 等 Python 指标）行为不变，通过 `_pre_next()` 钩子提前推进。
- 暖机期语义不变：最长 lookback 前只推进数据缓冲，不调用户 `next()`。
- 多 data feed primary clock 对齐语义不变，实现迁移至 Rust BarRunner。
- `compat.backtrader` 迁移策略零改动可运行。

### 关键实现约束

1. **BarRunner**：Rust struct，拥有 bar 迭代器，管理 primary/secondary feed 推进与
   tie-break 排序。
2. **共享 bar buffer**：Rust 写入当前 bar，Python `self.data[0]` 通过已有 property 读取，
   无需改动用户侧代码。
3. **`strategy._pre_next()`**：新增 Python 钩子，由 Rust 在调 `next()` 前触发，
   推进所有注册指标的 line cursor。此钩子不对用户可见，Strategy 基类内部维护。
4. **GIL 策略**：Rust 持有 GIL 完成整个 bar 处理周期（含两次 PyO3 回调），
   bar 间可释放 GIL。不得在持有 GIL 时执行阻塞 I/O。
5. **订单队列**：`next()` 中 `self.buy()` 产生的订单写入 Rust 侧队列，
   在下一根 bar `match_pending_orders()` 前撮合，语义与 1.0 一致。
6. **`callback_batch` 与 `trade_on_close`**：现有规则在 BarRunner 侧实现，
   Python 门面仅传参，不再控制循环节奏。

### 验收标准

- golden 50/50 compare 仍通过（trades 0 差异，PnL rtol=1e-4）。
- 现有全部单元测试与 golden 测试通过，无策略代码改动。
- benchmark（sma_cross + AAPL 10000 bars，20 次中位数）：
  - vs 1.0 Python 驱动循环 ≥ 3x
  - vs Backtrader oracle ≥ 8x
- `compat.backtrader` 十个迁移金标策略零改动运行通过。
