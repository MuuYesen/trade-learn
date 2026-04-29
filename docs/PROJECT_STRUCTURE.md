# 项目目录结构

本项目的核心定位是**事件驱动的量化研究、回测与后续实盘扩展工具**。目录划分应服务于三个边界:

- `tradelearn/core` 只放跨 backtest、paper、live 共同需要的契约和基础工具。
- `tradelearn/backtest` 直接承载回测运行时公共层,不再额外嵌套 `core/`。
- `tradelearn.engine` 与 `tradelearn.lite` 是唯二用户入口;不再维护 `compat` 迁移层。
- runner、benchmark、golden、example、archive 与本地 scratch 必须和运行时代码分开。

## 用户入口分层

trade-learn 提供两个用户面 API,两者底层共享同一套 Rust 回测引擎,回测结果在数值上一致,按使用者经验深度选择:

- **入门 / 快速验证**: `tradelearn.lite`
  风格对齐 Tradelearn 1.x Lite 语义。API 极简:`Strategy.init/next` + `self.buy/sell` + `self.I(...)`。
  适合策略原型、教学、单 data 场景、`pd.Series` 风格的快速指标输出。

- **资深 / 工程化**: `tradelearn.engine`
  风格对齐 Backtrader。API 完整:Analyzer / Observer / Sizer / Indicator / CommInfo / bracket orders / 多 data / multi-strategy 优化 / event 驱动 paper/live。
  适合复杂策略、组合、生产化部署、未来接入实盘。

> `tradelearn.backtest.*` 与 `tradelearn.core.*` 是上述两个 facade 的共享实现层与中性契约层,**不是**面向用户的公开 API,请勿直接 import。
> 未来 paper/live adapter 接入点在 `tradelearn.engine` 一侧,与现有 `Cerebro.run(mode="paper"|"live")` 路径一致。

## 架构总览

```
┌────────────────────────────────────────────────────────────────────────┐
│                       用户层 (User-Facing API)                         │
│                                                                        │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐    │
│   │ tradelearn.lite             │   │ tradelearn.engine           │    │
│   │                             │   │                             │    │
│   │   入门 / 快速验证           │   │   资深 / 工程化             │    │
│   │   • Strategy.init/next      │   │   • Cerebro / Strategy      │    │
│   │   • self.buy / self.sell    │   │   • Analyzer / Observer     │    │
│   │   • self.I(func, ...)       │   │   • Sizer / Indicator       │    │
│   │   • 单 data                 │   │   • bracket / OCO           │    │
│   │                             │   │   • 多 data / multi-strategy│    │
│   │                             │   │   • event 驱动 paper/live   │    │
│   └──────────────┬──────────────┘   └──────────────┬──────────────┘    │
└──────────────────┼─────────────────────────────────┼───────────────────┘
                   │   facade 可下行依赖             │
                   │   backtest 禁止反向依赖 facade  │
                   ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                共享回测运行时 (Private Runtime, 非公开 API)            │
│                       tradelearn.backtest                              │
│                                                                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│   │ engine       │  │ RustBroker   │  │ Strategy     │  │ EventRunner│ │
│   │ bar loop     │  │ broker proxy │  │ 内部基类     │  │ paper/live │ │
│   └──────┬───────┘  └──────┬───────┘  └──────────────┘  └────────────┘ │
│   ┌──────┴───────┐  ┌──────┴───────┐  ┌──────────────┐  ┌────────────┐ │
│   │ models       │  │ data         │  │ lines        │  │ indicator_ │ │
│   │ Order/Trade/ │  │ DataContainer│  │ LineSeries   │  │ cache      │ │
│   │ Position/... │  │ Bar buffers  │  │              │  │            │ │
│   └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────┬───────────────────────────────┬─────────────────┘
                       │ 通过 from_request/to_fill     │ 通过 PyO3 FFI
                       │ 与 core 中性契约互转          │
                       ▼                               ▼
        ┌────────────────────────────┐   ┌────────────────────────────┐
        │   中性契约层               │   │   Rust 高性能内核          │
        │   tradelearn.core          │   │   rust/tradelearn-rust/    │
        │                            │   │                            │
        │   • StreamBar              │   │   • matching.rs            │
        │   • OrderRequest / Ack     │   │     exact / smart 撮合     │
        │   • Fill                   │   │   • engine.rs              │
        │   • PositionSnapshot       │   │     RustBacktestEngine     │
        │   • AccountSnapshot        │   │     bar loop / portfolio   │
        │   • OrderStatusUpdate      │   │   • runner.rs              │
        │   • Broker Protocol        │   │     primary-clock          │
        │   • BrokerEventPump        │   │   • lib.rs (PyO3 bindings) │
        └────────────┬───────────────┘   └────────────────────────────┘
                     │
                     │  paper/live adapter 接入面 (未来扩展)
                     ▼
        ┌────────────────────────────┐
        │   tradelearn.brokers/*     │
        │   (QMT / IB / CTP / ...)   │
        │   — 当前为空,等真实 adapter│
        └────────────────────────────┘
```

### 关键数据流

回测路径(同步):

```
用户 Strategy.next()
   └→ self.buy/sell()
        └→ CoreStrategy.submit_order()  [backtest.strategy]
             └→ RustBroker._submit()    [backtest.broker]
                  └→ Rust submit_order  [via PyO3]
                       └→ matching.rs   [bar 内撮合]
                            └→ Fill 批量回流
                                 └→ broker._process_rust_fills_batch()
                                      └→ strategy.notify_trade / analyzer.on_fill
```

事件驱动路径(paper/live, 未来):

```
broker adapter (core.Broker Protocol)
   └→ emit Fill / OrderStatusUpdate (core.broker_contracts)
        └→ BrokerEventPump.dispatch()  [core.broker_events]
             └→ EventRunner.poll_broker_events()  [backtest.event_runner]
                  └→ Strategy.next() (统一事件驱动模型)
```

### 三条不可变量(在图上对应的位置)

1. **结果对齐** — 两条 facade 都走同一个 `RustBroker` + Rust matching,数值一致由共享内核保证。
2. **策略 API 清晰度** — 用户只面对最上层两个 facade 的 API,不暴露下层任何模块。
3. **paper/live 扩展边界** — `tradelearn.core` 中性契约 + `Broker` Protocol 是未来 broker adapter 的唯一接入面;`RustBroker` 走自己的 Rust 路径,不实现该 Protocol。

## 实操原则

Tradelearn 的架构在 v2 阶段已经基本成熟,后续功能开发与修改应考虑**优先维持当前架构**,性能层面优化排在第二地位。

### 架构定位

- **事件驱动为核心**:策略 API、指标推进、订单推进、portfolio 更新统一在事件驱动模型下表达,不引入与之冲突的执行模型。
- **Rust 只承担高性能内核**:撮合、bar loop、订单推进、portfolio 这四类高密度循环在 Rust。其他逻辑保持 Python。
- **指标计算不下沉 Rust**:通过 pandas-ta-classic、TDX、TradingView 等 Python 生态工具做批量缓存或 rolling 计算,避免在 Rust 维护指标公式。

### 目录边界(强制,与文件位置一一对应)

- `core/` 只放跨 backtest / paper / live 的中性契约。
- `backtest/` 只放公共回测 runtime,**不持有** Backtrader/backtesting.py 专属语义。
- `engine/` 与 `lite/` 各自维护对应 facade 的专属行为。
- 用户面入口仅 `engine/` 与 `lite/` 两条,`backtest/*` 与 `core/*` 不作为公开 API。

边界由 `tests/unit/backtest/test_core_layering.py` 在 CI 中守护,新增模块/重构时**禁止**绕开这些断言。

### 后续开发约束

- **优先级=性能优化**,默认假设是当前架构正确,先在不动边界的前提下做剖析和优化。
- **所有优化必须保留三条不可变量**:
  1. `tradelearn.engine` 与 Backtrader 的结果对齐(数值一致性);
  2. 策略 API 清晰度,不为跑分牺牲用户面易读性;
  3. 未来实盘扩展边界,即 core 中性契约的可演进性。
- **不为极限跑分牺牲清晰度**:微秒级提升若以策略 API 复杂化为代价,不接受。
- **不引入新的目录**或新的入口层,除非有明确的 paper/live adapter 接入需求并已经过设计评审。
- **不回流 facade 语义到 backtest/core**:即使为了性能或便利,也不允许把 `Params / TimeFrame / Backtrader Order 字段` 等 facade-only 类型上移。

## 运行时代码

### `tradelearn/backtest/`

共享回测核心,只放通用运行原语:

- bar loop / event loop (`tradelearn/backtest/engine.py`)
- broker-neutral models
- core strategy base
- Rust broker wrapper (`broker.py`)
- line primitives
- shared bar buffer
- indicator cache plumbing
- paper/live 可复用的事件 runner 和扩展接口

规则:

- 不允许 import `tradelearn.engine.*` 或 `tradelearn.lite.*`。
- 不放 Backtrader 或 backtesting.py 专属 API。
- 不导出 `Cerebro` / `Analyzer` / Backtrader `Strategy`;这些 facade 入口必须从 `tradelearn.engine` 使用。
- 不放具体 QMT、券商、交易柜台适配文件。
- 回测专属 runtime 不上移到 `tradelearn/core/`。

### `tradelearn/core/`

跨 backtest、paper、live 共用的基础设施:

- 配置、日志、错误类型、seed/time/progress
- `BrokerEvent` / `BrokerEventPump`
- 通用契约对象,例如 `StreamBar` 和 broker-neutral 的 `OrderRequest` / `Fill` / `PositionSnapshot`

它是更底层的运行基础,不应该知道任何 facade。

规则:

- 不允许 import `tradelearn.backtest.*`。
- 不允许 import `tradelearn.engine.*` 或 `tradelearn.lite.*`。
- 不接收回测专属 runtime,例如 bar loop、回测 broker wrapper、LineSeries、Sizer、Strategy 基类。
- 不接收回测状态机对象,例如 `Order`、`Trade`、`ExecutedInfo`; 这些只能由 runtime/facade 适配成 core 中性契约。
- 只放跨 backtest、paper、live 都成立的契约和基础工具。

### `tradelearn/engine/`

完整 API 层（原 Backtrader 风格）,维护 Backtrader 专属语义:

- `Cerebro`
- `Strategy`
- `PandasData`
- Backtrader metaclass/context 行为
- `params/self.p`
- Backtrader 风格 indicators / analyzers / observers / sizers
- Backtrader 风格 orders、bracket/OCO 元数据和 commission schemes
- `resampledata` / `replaydata` / `optstrategy` / `runstop` 等自动运行入口
- `plot` 占位入口和 `num2date` / `date2num` datetime helper
- `notify_order` / `notify_trade` helper

这里可以依赖 `tradelearn/backtest`,但 `tradelearn/backtest` 不允许反向依赖这里。

### `tradelearn/lite/`

入门 API 层（Tradelearn 1.x Lite 风格）,维护低级快速验证语义:

- `Backtest`
- `Strategy.I(...)`
- data proxy / indicator proxy / position proxy
- `data.close[0]` 当前 bar、`data.close[-1]` 前一根 bar
- `position()` / `position().close()`
- `pd.Series` 风格 stats 和参数优化行为

注意:

- `Strategy.I(...)` 是 Lite 专属 API,不应进入 core 或 engine facade。
- Lite 不兼容 backtesting.py 的 `self.data.Close[-1]` / `self.position.close()` 语法。
- 若需严格 Backtrader 行为与完整事件驱动 API,使用 `tradelearn.engine`。

## 测试分层原则

当前测试按“底层正确性”和“API 适配正确性”分层,避免重复验证同一件事:

| 层级 | 验收目标 | 代表测试 |
|---|---|---|
| `tradelearn.engine` | 验证共享回测 runtime、Rust 撮合、订单推进、portfolio 与 Backtrader 数值一致 | `benchmarks/runners/benchmark_bt.py` 必须保持 Backtrader `EXACT` |
| `tradelearn.lite` | 验证 Lite 语法能正确接入同一套 runtime | `tests/unit/lite/*`, `tests/unit/examples/test_1x_strategy_examples.py` |

因此:

- **底层正确性**由 `engine -> tradelearn.backtest runtime -> Rust` 对齐 Backtrader 负责。
- **Lite 测试只验证语法适配层**: `self.I(...)`、`data.close[0]`、`position()`、`I(DataFrame)`、`macd[:, 0]`、1.x 策略 smoke。
- Lite 不单独做逐笔撮合对齐,因为它没有自己的撮合/订单/portfolio 逻辑。
- Lite 不再对齐 `backtesting.py`;不引入 `from backtesting import ...`、`self.data.Close[-1]` 或 `self.position.close()` 作为正式测试入口。
- 验收组合是: **Lite smoke 通过 + `benchmark_bt.py` Backtrader EXACT**。

### `tradelearn/indicators/`

指标集成层:

- `core/`: pandas-ta-classic 相关指标封装和标准指标入口
- `tdx/`: TDX 指标族
- `tv/`: TradingView 风格指标族

规则:

- 指标公式不下沉到 Rust。
- core 只保留指标缓存/代理机制,不拥有具体指标公式。
- 回测可使用向量化预计算;实盘侧应使用 rolling window 增量/窗口重算。

### `tradelearn/data/`

数据规范化与缓存:

- Bars 契约
- provider adapter
- parquet cache / fingerprint / TTL / offline mode
- K 线重采样工具 (`resampler.py`)

### `tradelearn/factor/`, `tradelearn/metrics/`, `tradelearn/report/`, `tradelearn/ml/`

研究分析层:

- `factor/`: 因子分析、Alpha101/Alpha191
- `metrics/`: returns/risk/factor/trade 指标,以及回测 analyzer 复用的 `MetricsEngine`
- `report/`: HTML/Excel/Bokeh/pygwalker 报告
- `ml/`: MLStrategy、FeatureStore、ModelRegistry、CausalSelector

这些模块可以服务 backtest 输出,但不应把 facade 专属逻辑带回 core。

### `rust/tradelearn-rust/`

生产 Rust 执行核:

- `types.rs`: Order、Fill、Position、Bar、Portfolio 等公共结构。
- `matching.rs`: `exact` / `smart` K 线撮合逻辑。
- `engine.rs`: `RustBacktestEngine`、订单推进、bar loop、portfolio 更新。
- `runner.rs`: `RustBarRunner` / primary-clock / 多数据推进。
- `lib.rs`: PyO3 bindings 和对 Python 暴露的稳定 API。

Rust 负责高密度循环和撮合,Python 保留事件驱动策略 API 与生态兼容。

## 非运行时代码

### `examples/`

只放策略示例文件。

当前规则:

- `examples/backtrader/`: Backtrader facade 策略示例
- `examples/backtesting/`: backtesting.py facade 策略示例
- 不放 benchmark runner
- 不放 compare script
- 不放 CSV/parquet 数据
- 不放 shim 文件

布局测试由 `tests/unit/examples/test_examples_layout.py` 锁定。

### `benchmarks/`

性能基线、benchmark 数据和可执行 benchmark/profile runner:

- `baseline.json`
- migration blocker snapshot
- `data/backtesting/*.csv`
- `runners/benchmark_bt.py`
- `runners/compare_backtesting.py`
- `runners/speed_test_backtesting.py`
- `runners/profile_*.py`
- `runners/stress_benchmark.py`

规则:用于 benchmark/parity 的数据和性能 runner 放这里,不放 examples 或 tests。

### `scripts/examples/`

人工执行的示例、兼容诊断和迁移辅助脚本:

- `compat_test.py`
- `ml_strategy.py`
- `rf_fund.py`

这类脚本可用于人工诊断,但不是 pytest 断言本体。

### `tests/`

测试按语义分层:

- `tests/unit/`: 单模块快速测试
- `tests/consistency/`: 跨实现/跨 facade 一致性
- `tests/golden/`: golden manifest、策略、expected、returns
- `tests/smoke/`: 用户路径 smoke

规则:不放可执行 benchmark/profile/demo runner;这类脚本分别归入 `benchmarks/runners/` 或 `scripts/examples/`。

后续可优化:

- 将超大的 `test_rust_exact_matching.py` 拆成 matching / broker / line / multi-data 几组。
- 将 `test_alpha_metadata.py` 拆成 Alpha101 / Alpha191 / metadata validator。

### `docs/`

文档按用途分层:

- `docs/PROJECT.md`: 项目愿景、阶段规划、路线图
- `docs/PROJECT_STRUCTURE.md`: 当前目录边界和维护规则
- `docs/PROGRESS.md`: 当前状态摘要
- `docs/archive/`: 历史进度流水归档
- `docs/specs/`: 正式规格
- `docs/internal/`: 内部设计笔记
- `docs/release/`: 发布评估
- `docs/api/`: API reference 输出
- `docs/assets/`: 文档图片和静态资产

注意:`VISION.md` 已合并进 `PROJECT.md`,不再单独维护。

## 本地目录和生成物

以下内容不应提交:

- `.venv/`
- `target/`
- `.pytest_cache/`
- `.ruff_cache/`
- `reference/`
- `scratch/` 新增调试脚本
- `.idea/`
- `.vscode/`
- `.pypirc`
- `.rust_env/`
- `out.txt`
- `scratch.py`
- `test_quick.py`

## 当前仍可优化的目录语义

### 1. `tradelearn/query/` 可以继续收缩

当前 `query/` 主要是 1.x 兼容遗留和 MyTT 技术指标入口。长期看更清晰的方向是:

- 数据访问能力归入 `tradelearn/data/`
- 技术指标能力归入 `tradelearn/indicators/tdx/`
- `query/` 仅保留兼容 facade,或在 2.x 后续版本标记 deprecated

优先级:中。

### 2. `tradelearn/ml/automl/` 语义偏旧

`ml/automl/` 与当前 `MLStrategy / FeatureStore / ModelRegistry / CausalSelector` 主线不是同一层级语义。

建议:

- 若仍对外支持,明确写入 `ml/automl` 是 legacy/experimental。
- 若不再维护,迁到 `tradelearn/lab` 或后续 deprecate。

优先级:中低。

### 3. `tradelearn/brokers/` 目前应保持空或只放抽象接口

具体 QMT broker 不提交。后续若需要 broker 目录,建议只放:

- `base.py`
- `events.py`
- `adapter.py`

具体实现如 QMT、IB、CTP 应进入插件或私有扩展目录。

优先级:低,等实盘接口正式推进再做。

### 4. Alpha 大文件可以拆,但不是当前风险

- `tradelearn/factor/alpha/alpha191.py`
- `tradelearn/factor/alpha/alpha101.py`

这些文件大,但语义集中。拆分只有在维护成本明显上升时再做。

优先级:低。

### 5. `docs/assets/` 可后续压缩

图片资产体积较大,但不影响运行时代码边界。若发布包或文档站体积成为问题,再考虑压缩或迁到外部资产。

优先级:低。

## 边界检查清单

提交前建议检查:

```bash
grep -R "tradelearn.engine\|tradelearn.lite" tradelearn/backtest tradelearn/core
grep -R "tradelearn.backtest" tradelearn/core
uv run pytest tests/unit/examples/test_examples_layout.py -q
uv run pytest tests/unit/backtest/test_core_layering.py -q
uv run pytest tests/unit/docs/test_docs_completeness.py -q
```

目录语义判断:

- 策略示例进 `examples/`
- 可执行 benchmark/parity/profile 进 `benchmarks/runners/`
- 人工 demo/compat runner 进 `scripts/examples/`
- benchmark 数据进 `benchmarks/data/`
- 当前状态进 `docs/PROGRESS.md`
- 历史流水进 `docs/archive/`
- facade 专属行为进 `tradelearn/engine/` 或 `tradelearn/lite/`
- 回测事件驱动 runtime 进 `tradelearn/backtest`
- 跨 backtest/paper/live 契约进 `tradelearn/core`
