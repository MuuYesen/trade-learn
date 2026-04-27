# MIGRATION

trade-learn 2.0 相对 1.x 的差异说明 + Known Differences 登记簿。

## 1. 背景

- trade-learn 2.0 是对 1.x 的完整重构
- **1.x 无现有用户**,不保留 API 兼容
- 本文档作用:**记录变更** + **Known Differences 追溯**,而非"迁移指南"

## 2. 主要变更概览(1.x → 2.0)

### 2.1 架构层

| 方面 | 1.x | 2.0 |
|---|---|---|
| 撮合引擎 | Python(基于 backtesting.py) | Rust(Clean-Room 参考 backtesting.py) |
| 用户 API | backtesting.py 风格 | **严格 backtrader 风格** |
| Vendor 策略 | 内嵌 pyfolio/alphalens/empyrical/pandas_ta 源码 | 融合进 metrics/factor/report,vendor 全清 |
| 数据源 | yfinance + 旧 TDX provider + tvdatafeed | 剔除 yfinance 和旧 TDX 依赖;统一为 opentdx + TV |
| 协议 | 未明确 | **Apache-2.0** |

### 2.2 API 层

| 1.x | 2.0 |
|---|---|
| `from tradelearn.query import Query` | `from tradelearn.data import Query` |
| `Backtest(data, Strategy).run()` | `cerebro = Cerebro(); cerebro.adddata; cerebro.addstrategy; cerebro.run()` |
| `class S(Strategy): def init(self)` | `class S(Strategy): def __init__(self)` |
| `self.data.close[-1]`(当前) | `self.data.close[0]`(当前)**[0] ≠ -1** |
| 类属性 `fast = 10` | `params = (('fast', 10),)` |
| `self.I(func, ...)` 注册 | 直接 `self.ma = ta.sma(...)` |
| `Evaluate.analysis_report(stats)` | `Reporter(stats).html()` |
| `from tradelearn.strategy.examine import ...` | `from tradelearn.factor import FactorAnalyzer` |
| `from tradelearn.strategy.evaluate import ...` | `from tradelearn.metrics import sharpe, ...` |
| `AutoML` / `CausalGraph` | `from tradelearn.ml import MLStrategy, CausalSelector` |

### 2.3 指标层

| 1.x | 2.0 |
|---|---|
| `from tradelearn.query.tec import ...` | `from tradelearn import ta` |
| `pandas_ta` vendor | `ta.*` → pandas-ta-classic 封装 |
| tdx30 分散 | `ta.tdx.*`(MyTT 为算法源) |
| 无外盘专用 | `ta.tv.*`(pyneCore 后端) |

### 2.4 追踪 / 报告

| 1.x | 2.0 |
|---|---|
| 手动 pyfolio / quantstats | `Reporter(stats).html()` |
| 无 MLflow 集成 | `cerebro.addanalyzer(MLflowAnalyzer, ...)` |
| `stats.plot()` | `cerebro.plot()`(bokeh) |

### 2.5 项目结构

| 1.x | 2.0 |
|---|---|
| `tradelearn/query/tec/pandas_ta/` | 删除 |
| `tradelearn/strategy/evaluate/pyfolio/` | 删除(融合进 report) |
| `tradelearn/strategy/evaluate/empyrical/` | 删除(融合进 metrics) |
| `tradelearn/strategy/examine/alphalens/` | 删除(融合进 factor) |
| `tradelearn/causal/graph/causallearn/` | 删除(改 pip 依赖 causallearn) |
| — | 新增 `tradelearn/mcp/` |
| — | 新增 `tradelearn/lab/` |
| — | 新增 `backtest-rs/` |
| — | 新增 `tradelearn/compat/backtrader/` |

### 2.6 依赖

| 1.x | 2.0 |
|---|---|
| `yfinance` | ❌ 移除 |
| `旧 TDX provider` | ❌ 替换为 `opentdx` |
| 无 MLflow | ✅ `mlflow`(核心) |
| 无 MCP | ✅ `mcp`(核心) |
| 无 JupyterLab | ✅ `[lab]` extras |

## 3. Known Differences(登记簿)

所有新旧版本**可接受的、可解释的差异**在此登记。**任何未登记的差异都是 bug**。

### 3.1 条目格式

```markdown
### [日期] 差异标题

**位置**:`tradelearn.xxx.yyy`
**影响**:一句话说清楚谁会遇到
**原因**:为什么会有差异(算法 / 精度 / 实现方式)
**影响评估**:对用户决策的实际影响(通常是"无" / "数值 < X%")
**测试**:对应的 consistency test 和放宽后的 rtol
```

### 3.2 登记示例(待填充)

### Stage 3 Week 5 golden / migration blockers

| ID | Status | Ready | Total | Reason | Next |
|---|---|---:|---:|---|---|
| `golden-datasets` | blocked | 0 | 5 | TDX 永久放弃(海外无法访问),0.1-alpha golden 仅以 TV subset 为目标;缺少 tracked TV subset parquet artifacts | materialize TV subset parquet files under `tests/golden/data/tv` |
| `backtrader-oracle-runner` | ready | 3 | 3 | `scripts/run_backtrader_oracle.py` 与 `build_golden.py --oracle backtrader` 已建立 SMA/MACD/KDJ 最小 Backtrader parity smoke | reviewer 确认后用于生成 TV subset Backtrader expected artifacts |
| `golden-expected-v1` | blocked | 0 | 50 | TV subset expected 生成路径已可用,Backtrader parity runner 最小 smoke 已就绪,但真实 TV parquet 与 Backtrader expected artifacts 尚未冻结 | TV subset parquet 落盘后生成 Backtrader expected artifacts |
| `golden-strategy-adapters` | accepted | 10 | 10 | 10 个 golden strategy adapter 已可作为 Stage 3 proxy adapter 运行 | full 1.0 parity 前按 Backtrader 示例语义替换/补齐最终策略经济语义 |
| `full-golden-comparison` | blocked | — | — | `scripts/compare_golden.py --engine tv` 对照门禁已就绪,但 TV subset trades 零差异与 PnL `rtol=1e-4` 真正验收仍依赖 Backtrader oracle expected/v1.0 artifacts | TV subset Backtrader oracle expected + `scripts/check_golden_readiness.py --engine tv` 返回 ok=true 后执行全量金标对照 |

机器可读快照:`benchmarks/stage3_migration_blockers.json`
校验命令:`uv run python scripts/check_stage3_migration.py --json`

### Stage 3 deferred semantics

| ID | Status | Reason | Impact | Next |
|---|---|---|---|---|
| `stats-summary-sharpe-max-dd` | resolved | `Stats.summary` 已接入 metrics 层 `sharpe` / `max_drawdown`;MLflow metrics 只记录有限数值,artifact 保留完整 summary | Report / MLflow 可追踪基础 cash/equity/PnL 与核心风险摘要;常量收益 Sharpe 为 NaN 时不写入 MLflow metrics | Reviewer 9722b72 已确认可标为 resolved;后续若扩充 16 项 report summary,另行登记 |
| `auto-min-period-inference` | deferred | 当前 `Strategy.addminperiod()` / 类属性 `min_period` 已支持手动暖机期,尚未扫描指标对象自动推导最长 lookback | 策略可显式声明暖机期;未声明时不会自动从指标对象推断 | 指标 Line 对象契约稳定后补自动 lookback 收集与事件循环测试 |

### Alpha alpha101 skipped formulas

| Formula | Reason |
|---|---|
| `alpha048` | requires industry neutralization input |
| `alpha056` | requires cap input |
| `alpha058` | requires industry neutralization input |
| `alpha059` | requires industry neutralization input |
| `alpha063` | requires industry neutralization input |
| `alpha067` | requires industry neutralization input |
| `alpha069` | requires industry neutralization input |
| `alpha070` | requires industry neutralization input |
| `alpha076` | requires industry neutralization input |
| `alpha079` | requires industry neutralization input |
| `alpha080` | requires industry neutralization input |
| `alpha082` | requires industry neutralization input |
| `alpha087` | requires industry neutralization input |
| `alpha089` | requires industry neutralization input |
| `alpha090` | requires industry neutralization input |
| `alpha091` | requires industry neutralization input |
| `alpha093` | requires industry neutralization input |
| `alpha097` | requires industry neutralization input |
| `alpha100` | requires subindustry neutralization input |

### Alpha alpha191 skipped formulas

| Formula | Reason |
|---|---|
| `alpha030` | requires external MKT/SMB/HML regression inputs |
| `alpha075` | legacy formula is commented and requires benchmark condition counts |
| `alpha143` | legacy formula is commented placeholder |
| `alpha149` | requires benchmark filter input |
| `alpha181` | legacy formula is commented and requires benchmark close input |
| `alpha182` | legacy formula is commented and requires benchmark open/close input |
| `alpha190` | legacy formula is commented placeholder |


---

#### 模板条目(示例,尚未实际发生)

```
### [2026-05-15] ta.tdx.MACD 初始值差 1e-8

**位置**:`tradelearn.indicators.tdx.MACD`
**影响**:`ta.tdx.MACD(close)['DIF']` 前 12 根数值和 MyTT 原版差 1e-8
**原因**:EMA 初始化用 SMA 引导(backtesting.py 习惯),MyTT 用第一个值
**影响评估**:前 12 根暖机期的数值差异,信号时点无变化,累计后差 < 1e-10
**测试**:`tests/consistency/test_indicators_tdx.py::test_macd[...]`
       对 MyTT 放宽 rtol 从 `1e-10` 到 `1e-8`
```

## 4. 数据路径变更

| 旧路径(1.x) | 新路径(2.0) |
|---|---|
| 隐式(每次重新拉)| `./data/{engine}/{symbol}_{range}.parquet` |
| — | `~/.cache/tradelearn/`(全局缓存,可选) |

## 5. 配置变更

| 旧 | 新 |
|---|---|
| 无环境变量 | `MLFLOW_TRACKING_URI=https://mlflow.leafquant.com` |
| 无 | `TRADELEARN_DATA_CACHE_DIR=./data` |
| 无 | `TRADELEARN_LOG_LEVEL=INFO` |
| 无 | `TRADELEARN_SEED=42`(可选 reproducibility) |

## 6. 报告格式变更

| 1.x | 2.0 |
|---|---|
| 依赖原 pyfolio HTML | 自带 HTML 模板(jinja2 + bokeh) |
| matplotlib 图 | 全部 bokeh 交互图 |
| 无 Excel 导出 | `Reporter(stats).excel()` |
| 无交互探索 | `Reporter(stats).explore()`(pygwalker) |

## 7. 1.0 后兼容策略

- 1.0 发版 = API 冻结
- 1.x 的 0.x.y patch 不 break API
- 2.0 允许 break(走 breaking change 流程,本文档再追加大版本)
- 关键 API 变动必须:
  - MIGRATION.md 记录
  - DeprecationWarning 至少一个 minor 版本
  - Release Notes 显著标注

## 8. 如何查 Known Differences

### 用户遇到"新旧结果不一致"时

1. 先查本文档 § 3.2
2. 若有条目 → 了解原因,可接受
3. 若无条目 → 报 issue,视为 bug

### 开发者添加新条目

1. PR 修改金标 rtol 时,必须同时在 MIGRATION.md 加条目
2. 条目必须填齐 4 个字段(位置 / 原因 / 影响 / 测试)
3. 走独立 PR,两人 review

## 9. 1.x 归档

- 1.0 发版后,1.x 源码归档到 GitHub Release("Legacy 1.x")
- `reference/tradelearn_1x/` 可从主仓删除(已在 GitHub Release 有副本)

## 10. 不做的事

- ❌ 保留 1.x 兼容 shim(没人用,浪费)
- ❌ 自动化"1.x → 2.0 代码转换器"(无 ROI)
- ❌ 在 2.0 里同时暴露两套 API(混乱)
- ❌ 因为"看着像老用户可能需要"就加个 deprecated 别名
