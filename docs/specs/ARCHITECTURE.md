# ARCHITECTURE

trade-learn 2.0 系统架构规格。

## 1. 分层架构

```
┌──────────────────────────────────────────────────────────┐
│  用户层(Python API,严格 backtrader 风格)               │
│  Cerebro / Strategy / Analyzer / Query / ta.*            │
├──────────────────────────────────────────────────────────┤
│  应用层                                                   │
│  backtest · factor · report · ml · compat.backtrader     │
├──────────────────────────────────────────────────────────┤
│  核心能力层                                               │
│  indicators · metrics · brokers · tracking(内部)        │
├──────────────────────────────────────────────────────────┤
│  契约层(core/)                                          │
│  Bars / Factor / Signal / Returns / StreamBar /          │
│  Experiment / Broker Protocol                            │
├──────────────────────────────────────────────────────────┤
│  数据与引擎                                               │
│  data(opentdx/tvdatafeed)· Rust 撮合核(_core.so)      │
└──────────────────────────────────────────────────────────┘

周边:
  lab · mcp · cli(工具与集成,对上 6 层只读调用)
```

## 2. 模块清单

| 模块 | 职责 | 外部依赖 | 内部依赖 |
|---|---|---|---|
| `core/` | 7 个公共契约对象 | — | — |
| `data/` | 行情拉取 + 缓存 | opentdx / tvdatafeed / pyarrow | core |
| `indicators/` | `ta.*` / `ta.tdx.*` / `ta.tv.*` | pandas-ta-classic / pynecore | core |
| `metrics/` | 指标唯一真源(融合 empyrical) | numpy / pandas / scipy | core |
| `factor/` | 因子评估(融合 alphalens) | numpy / pandas | core / metrics |
| `report/` | 策略报告(融合 pyfolio + quantstats) | bokeh / xlsxwriter | core / metrics |
| `backtest/` | Cerebro + Strategy + Analyzer + Rust 核(1.1 新增 RustBarRunner) | PyO3 / arrow | core / metrics / report |
| `backtest/analyzers/` | Analyzer 体系(含 MLflowAnalyzer) | mlflow | core / metrics / report |
| `ml/` | MLStrategy + Feature Store + causal | scikit-learn / causallearn | core / backtest |
| `brokers/` | 实盘 Broker(1.1:QMTBroker) | httpx | core / backtest |
| `compat/backtrader/` | backtrader API 兼容层 | — | backtest |
| `mcp/` | docstring 暴露给 LLM | mcp | 所有(只读) |
| `lab/` | CLI + 模板 | typer | 所有(只 invoke) |

## 3. 依赖方向规则

**严格单向**:上层可调下层,下层不得调上层。

```
lab ────► mcp ──┐
                ├──► compat.backtrader ──► backtest ──┬──► factor ──► metrics ──► core
ml ─────► backtest ─────────────────────────────────┤
                                          backtest ──┴──► report ───► metrics
indicators ─► core
data ──────► core
brokers ───► backtest
```

### 硬规则

- **`core/` 不依赖任何其它模块**
- **`metrics/` 不依赖任何上层**(是指标唯一真源)
- 循环依赖禁止(CI 用 `pydeps` 或 `importlab` 强制)
- 同层之间默认不互调(例外:`factor/` 调 `metrics/`,`report/` 调 `metrics/`)

## 4. 跨语言边界

```
Python 侧(tradelearn/backtest/__init__.py)
      │ PyO3 binding
      ▼
Rust 侧(backtest-rs/)
  - engine      事件循环
  - matcher     订单撮合
  - portfolio   记账
  - events      事件类型
```

- Python ↔ Rust **通过 Apache Arrow zero-copy**
- Rust 产物:`backtest-rs/` 编译的 `_core.*.so`(由 maturin 安装到 `tradelearn/backtest/`)
- 策略逻辑(`Strategy.__init__ / next / notify_*`)**留在 Python**,Rust 侧只做撮合/记账

## 5. 命名空间规则

### Python import 路径

```python
# 核心用户 API
from tradelearn.backtest import Cerebro, Strategy, Analyzer
from tradelearn.backtest.analyzers import MLflowAnalyzer
from tradelearn.data import Query
from tradelearn import ta                         # ta.sma, ta.rsi
from tradelearn.indicators import tdx, tv          # 或直接 ta.tdx / ta.tv

# 能力扩展
from tradelearn.metrics import sharpe, max_drawdown
from tradelearn.factor import FactorAnalyzer
from tradelearn.report import Reporter
from tradelearn.ml import MLStrategy, CausalSelector

# 迁移/兼容
import tradelearn.compat.backtrader as bt
```

### `ta` 命名空间暴露

```python
# tradelearn/__init__.py
from tradelearn.indicators import ta
```

三子命名空间:
- `ta.*` = pandas-ta-classic 通用
- `ta.tdx.*` = 通达信口径(算法源 MyTT)
- `ta.tv.*` = TradingView 口径(pyneCore 后端)

同名指标口径不同,由用户显式选择。

## 6. 扩展点

### 用户层(继承 / 函数)

| 扩展点 | 方式 | 示例 |
|---|---|---|
| Strategy | 继承 `Strategy` | `class MyStrategy(Strategy): ...` |
| Indicator | 普通函数 | `def my_ind(close, n): return ...` |
| Analyzer | 继承 `Analyzer` | `class MyAnalyzer(Analyzer): on_start / on_end` |
| MLStrategy | 继承 `MLStrategy` | 自定义 model / features / target |

### 框架层(Protocol 实现)

| 扩展点 | 协议 | 用途 |
|---|---|---|
| Broker | `core.Broker` | 自定义券商/撮合 |
| DataFeed | `core.DataFeed` | 自定义数据源 |
| Reporter | `core.Reporter` | 自定义报告输出 |
| FeatureStore | `ml.FeatureStore` | 自定义因子仓库后端 |

### 核心层(不开放)

- Rust 撮合引擎(`_core.so` 黑盒)
- `metrics/` 函数(一致性基线,不允许替换)
- 事件循环语义

## 7. 配置与环境

### 优先级(按高到低)

1. Python API 显式参数(`cerebro.addanalyzer(MLflowAnalyzer, uri="...")`)
2. 环境变量(`MLFLOW_TRACKING_URI` / `TRADELEARN_DATA_CACHE_DIR` 等)
3. 项目 `.tradelearn/config.yaml`(可选)
4. 用户 `~/.tradelearn/config.yaml`(可选)
5. 默认值(代码内)

### 标准环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `https://mlflow.leafquant.com` | MLflow server |
| `TRADELEARN_DATA_CACHE_DIR` | `./data` | 数据缓存目录 |
| `TRADELEARN_LOG_LEVEL` | `INFO` | 日志级别 |
| `TRADELEARN_SEED` | unset | 全局随机种子(可复现) |

## 8. 非功能性要求

### 性能目标(Rust 撮合核)

| 场景 | 目标 |
|---|---|
| 单品种 10 年日线 | < 50 ms |
| 500 股组合 10 年日线 | < 5 s |
| 流式模式单根 bar | < 1 ms(含 Python callback) |

### 跨平台

- OS:Linux / macOS / Windows
- Python:3.10 / 3.11 / 3.12
- 发布物:跨平台 wheel(cibuildwheel)

**1.0 跨平台严守;实盘 `[live-qmt]` 仅 Windows。**

### 可观测

- 统一 logging(Python `logging` + 结构化)
- tqdm 进度条(长任务)
- Rust 侧日志通过 callback 回 Python logger

### 确定性

- Rust 侧使用 `BTreeMap`(不是 `HashMap`)保证遍历顺序
- 事件队列稳定排序(同时刻按提交顺序)
- 浮点累积顺序固定(不使用并行 reduce)
- 全局 seed 支持

## 9. 不做的事(架构层面)

- ❌ Pipeline DSL 作为一等公民(降级为辅助工具,不进主架构)
- ❌ 插件自动注册(entry_points)——1.0 显式注册
- ❌ 多进程调度器——grid_search 等在应用层做,不进核心
- ❌ Web UI / HTTP 服务端(1.0 是纯 SDK)
- ❌ Tick 级事件循环(1.0 只 bar 级)

## 10. 文档工程

- docstring 为唯一源头(Numpy-style)
- `mkdocs-material` + `mkdocstrings-python` 自动生成 API 页
- `interrogate --fail-under 90` 强制覆盖率
- `pytest --doctest-modules` 保证示例可跑
- `mike` 版本化(v1.0 / v1.1 并存)

## 11. 协议与署名

- 主协议:**Apache-2.0**
- NOTICE 文件必须列出:empyrical / alphalens / pyfolio / quantstats / MyTT / DolphinDB / causallearn / pandas-ta-classic / pyneCore
- backtesting.py / backtrader 标注"inspired by"(不复制源码)
