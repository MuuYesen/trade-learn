# 项目目录结构

本项目的核心定位是**事件驱动的量化研究、回测与后续实盘扩展工具**。目录划分应服务于三个边界:

- `core` 只放所有模式共同需要的事件、数据、撮合、缓存和扩展接口。
- `compat` 只放迁移层,分别维护 Backtrader 和 backtesting.py 的专属行为。
- runner、benchmark、golden、example、archive 与本地 scratch 必须和运行时代码分开。

## 运行时代码

### `tradelearn/backtest/core/`

共享回测核心,只放通用运行原语:

- bar loop / event loop
- broker-neutral models
- core strategy base
- Rust broker wrapper (`broker.py`)
- line primitives
- shared bar buffer
- indicator cache plumbing
- paper/live 可复用的事件 runner 和扩展接口

规则:

- 不允许 import `tradelearn.compat.*`。
- 不放 Backtrader 或 backtesting.py 专属 API。
- 不放具体 QMT、券商、交易柜台适配文件。
- 回测专属 runtime 不上移到 `tradelearn/core/`。

### `tradelearn/core/`

跨 backtest、paper、live 共用的基础设施:

- 配置、日志、错误类型、seed/time/progress
- `BrokerEvent` / `BrokerEventPump`
- 通用契约对象

它是更底层的运行基础,不应该知道任何 facade。

规则:

- 不允许 import `tradelearn.backtest.*`。
- 不允许 import `tradelearn.compat.*`。
- 不接收回测专属 runtime,例如 bar loop、回测 broker wrapper、LineSeries、Sizer、Strategy 基类。
- 只放跨 backtest、paper、live 都成立的契约和基础工具。

### `tradelearn/compat/backtrader/`

Backtrader 迁移层,维护 Backtrader 专属语义:

- `Cerebro`
- `Strategy`
- `PandasData`
- Backtrader metaclass/context 行为
- `params/self.p`
- Backtrader 风格 indicators / analyzer / sizer
- `notify_order` / `notify_trade` helper

这里可以依赖 `backtest/core`,但 core 不允许反向依赖这里。

### `tradelearn/compat/backtesting/`

backtesting.py 迁移层,维护 backtesting.py 专属语义:

- `Backtest`
- `Strategy.I(...)`
- data proxy / indicator proxy / position proxy
- backtesting.py 风格 stats 和参数优化行为

注意:`Strategy.I(...)` 是 backtesting.py facade 专属 API,不应进入 core 或 Backtrader facade。

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

- K 线撮合
- portfolio / position / fill/order 语义
- Rust BarRunner / primary-clock plan
- PyO3 bindings

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

### `tests/runners/`

可执行 runner、benchmark、profile 和 parity 脚本:

- `benchmark_bt.py`
- `compare_backtesting.py`
- `speed_test_backtesting.py`
- stress/profile/compat runner

这类脚本可以被 CI 或人工执行,但不属于 examples。

### `benchmarks/`

性能基线和 benchmark 数据:

- `baseline.json`
- migration blocker snapshot
- `data/backtesting/*.csv`

规则:用于 benchmark/parity 的数据放这里,不放 examples。

### `tests/`

测试按语义分层:

- `tests/unit/`: 单模块快速测试
- `tests/consistency/`: 跨实现/跨 facade 一致性
- `tests/golden/`: golden manifest、策略、expected、returns
- `tests/smoke/`: 用户路径 smoke
- `tests/runners/`: 可执行测试脚本

后续可优化:

- 将超大的 `test_rust_exact_matching.py` 拆成 matching / broker / line / multi-data 几组。
- 将 `test_strategy_api.py` 拆成 order lifecycle / portfolio / analyzer / multi-data。
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
grep -R "tradelearn.compat" tradelearn/backtest/core tradelearn/core
grep -R "tradelearn.backtest" tradelearn/core
uv run pytest tests/unit/examples/test_examples_layout.py -q
uv run pytest tests/unit/backtest/test_core_layering.py -q
uv run pytest tests/unit/docs/test_docs_completeness.py -q
```

目录语义判断:

- 策略示例进 `examples/`
- 可执行 benchmark/parity 进 `tests/runners/`
- benchmark 数据进 `benchmarks/data/`
- 当前状态进 `docs/PROGRESS.md`
- 历史流水进 `docs/archive/`
- facade 专属行为进 `tradelearn/compat/*`
- 回测事件驱动 runtime 进 `tradelearn/backtest/core`
- 跨 backtest/paper/live 契约进 `tradelearn/core`
