# v1 → v2 迁移

trade-learn 2.0 是对 1.x 的**完整重构**。1.x 没有现存用户基础，因此 2.0 不保留 API 兼容；本页的作用是说明**变更内容**和**已知可接受差异**的登记簿，而非"渐进迁移指南"。

> 如果你是新用户，直接读 [快速开始](../quickstart.md) 即可，不用关心本页。本页面向：曾经接触过 1.x 代码、维护 1.x fork、或在阅读历史 commit 时遇到旧 API 的读者。

## 1. 主要变更概览

### 1.1 架构层

| 方面 | 1.x | 2.0 |
|---|---|---|
| 撮合引擎 | Python（基于 backtesting.py） | Rust（Clean-Room，参考 backtesting.py 概念） |
| 用户 API | backtesting.py 风格 | **严格 backtrader 风格** |
| Vendor 策略 | 内嵌 pyfolio / alphalens / empyrical / pandas_ta 源码 | 融合进 metrics / factor / report，vendor 全清 |
| 数据源 | yfinance + 旧 TDX provider + tvdatafeed | 移除 yfinance 与旧 TDX，统一为 **opentdx + TV** |
| 协议 | 未明确 | **Apache-2.0** |

### 1.2 用户 API

| 1.x | 2.0 |
|---|---|
| `from tradelearn.query import Query` | `import tradelearn.factor as tf` / `import tradelearn.indicators as ta` / `tradelearn.data` |
| `Backtest(data, Strategy).run()` | `cerebro = Cerebro(); cerebro.adddata; cerebro.addstrategy; cerebro.run()` |
| `class S(Strategy): def init(self):` | `class S(Strategy): def __init__(self):` |
| `self.data.close[-1]`（当前 bar） | `self.data.close[0]`（当前 bar）—— **`[0]` 是当前，不是 `[-1]`** |
| 类属性 `fast = 10` | `params = (('fast', 10),)` |
| `self.I(func, ...)` 注册指标 | 直接 `self.ma = ta.sma(...)` |
| `from tradelearn.strategy.evaluate import Evaluate` | `from tradelearn.report import Reporter` |
| `from tradelearn.causal.graph import Graph` | `import tradelearn.ml as ml` |
| `from tradelearn.automl import AutoML` | `import tradelearn.ml as ml` |

### 1.3 指标层

| 1.x | 2.0 |
|---|---|
| `Query.tec_indicator(...)` | `import tradelearn.indicators as ta` |
| 内嵌 `pandas_ta` vendor | `tl.pta` / `bt.pta` → pandas-ta-classic 封装 |
| `tdx30` 分散指标 | `tl.tdx` / `bt.tdx`（MyTT 算法源） |
| 无外盘专用 | `tl.tv` / `bt.tv`（pyneCore 后端） |

### 1.4 报告与追踪

| 1.x | 2.0 |
|---|---|
| 手动调用 pyfolio / quantstats | `Reporter(stats).report("report.html")` |
| 无 MLflow 集成 | `cerebro.addanalyzer(MLflowAnalyzer, ...)` |
| `stats.plot()` matplotlib | `cerebro.plot()` bokeh 交互图 |
| 无 Excel 导出 | `Reporter(stats).report("report.xlsx")` |
| 无交互探索 | `Reporter(stats).explore()`（pygwalker） |

### 1.5 项目结构

| 1.x 路径 | 2.0 |
|---|---|
| `tradelearn/query/` | 删除（职责拆入 `factor` / `indicators` / `data`） |
| `tradelearn/strategy/` | 删除（拆入 `engine` / `lite` / `report` / `metrics` / `factor`） |
| `tradelearn/strategy/evaluate/pyfolio/` | 删除（融合进 `report`） |
| `tradelearn/strategy/evaluate/empyrical/` | 删除（融合进 `metrics`） |
| `tradelearn/strategy/examine/alphalens/` | 删除（融合进 `factor`） |
| `tradelearn/causal/graph/causallearn/` | 删除（改为 pip 依赖 `causallearn`） |
| — | 新增 `tradelearn/mcp/` |
| — | 新增 `tradelearn/lab/` |
| — | 新增 `backtest-rs/`（Rust 内核） |
| — | 新增 `tradelearn/compat/backtrader/` |

### 1.6 依赖

| 1.x | 2.0 |
|---|---|
| `yfinance` | ❌ 移除 |
| 旧 TDX provider | ❌ 替换为 `opentdx` |
| 无 MLflow | ✅ `mlflow`（核心） |
| 无 MCP | ✅ `mcp`（核心） |
| 无 JupyterLab | ✅ `[lab]` extras |

### 1.7 数据缓存路径

| 1.x | 2.0 |
|---|---|
| 隐式（每次重新拉） | `./data/{engine}/{symbol}_{range}.parquet` |
| — | `~/.cache/tradelearn/`（全局缓存，可选） |

### 1.8 配置

新增四个标准环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `https://mlflow.leafquant.com` | MLflow server |
| `TRADELEARN_DATA_CACHE_DIR` | `./data` | 数据缓存目录 |
| `TRADELEARN_LOG_LEVEL` | `INFO` | 日志级别 |
| `TRADELEARN_SEED` | unset | 全局随机种子（可复现） |

## 2. 1.0 后的兼容策略

- **1.0 发版 = API 冻结**
- 1.x 的 `0.x.y` patch 不 break API
- 2.0 允许 break，需要走 breaking change 流程：
  - 本页 Known Differences 登记
  - `DeprecationWarning` 至少一个 minor 版本
  - Release Notes 显著标注

## 3. Known Differences（登记簿）

**任何未登记的差异都视为 bug**。当前登记内容如下。

### 3.1 条目格式

```
### [日期] 差异标题

位置：tradelearn.xxx.yyy
影响：一句话说清楚谁会遇到
原因：为什么会有差异（算法 / 精度 / 实现方式）
影响评估：对用户决策的实际影响（通常是"无" / "数值 < X%"）
测试：对应的 consistency test 与放宽后的 rtol
```

### 3.2 alpha101 跳过的公式

以下公式因依赖外部输入（行业中性化、市值），目前**未实现**：

`alpha048` `alpha056` `alpha058` `alpha059` `alpha063` `alpha067` `alpha069` `alpha070` `alpha076` `alpha079` `alpha080` `alpha082` `alpha087` `alpha089` `alpha090` `alpha091` `alpha093` `alpha097` `alpha100`

### 3.3 alpha191 跳过的公式

以下公式依赖外部 MKT/SMB/HML 回归输入或 benchmark 条件，**未实现**：

`alpha030` `alpha075` `alpha143` `alpha149` `alpha181` `alpha182` `alpha190`

### 3.4 模板示例（尚未实际触发）

```
### [2026-05-15] tl.tdx.MACD 初始值差 1e-8

位置：tradelearn.indicators.tdx.MACD
影响：tl.tdx.MACD(close)['DIF'] 前 12 根数值和 MyTT 原版差 1e-8
原因：EMA 初始化用 SMA 引导，MyTT 用第一个值
影响评估：前 12 根暖机期数值差异，信号时点无变化，累计后差 < 1e-10
测试：tests/consistency/test_indicators_tdx.py::test_macd[...]
     对 MyTT 放宽 rtol 从 1e-10 到 1e-8
```

## 4. 用户遇到"新旧结果不一致"时

1. 先查本页 § 3
2. 若有条目 → 了解原因，差异可接受
3. 若无条目 → 报 issue，视为 bug

## 5. 1.x 归档

- 1.0 发版后，1.x 源码归档到 GitHub Release（"Legacy 1.x"）
- 主仓的 `reference/tradelearn_1x/` 在 1.0 发版后可移除（GitHub Release 已有副本）

## 相关阅读

- [与外部库的语义一致性](consistency.md)：分层容忍度与金标对照
- [契约与边界](contracts.md)：2.0 的契约对象口径
