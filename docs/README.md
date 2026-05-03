# trade-learn 技术手册

<p align="center">
  <img src="tradelearn-logo.png" alt="trade-learn logo" width="520" />
</p>

**trade-learn** 是一个面向量化研究、机器学习策略与事件驱动回测的 Python / Rust 框架。

- **Python** 表达策略、因子、模型、研究流程
- **Rust** 承担撮合、订单推进、bar runner、portfolio 这类高频回测内核
- **共享同一套契约和 Stats**——研究、回测、报告之间不需要再做格式转换

## 设计目标

trade-learn 的目标不是再写一个回测框架，而是把一条**完整的策略研发链路**接起来：

<p align="center">
  <img src="research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

数据 → 指标 → 因子 → 探索 / 因果分析 → 模型 → 组合权重 → 回测 → 报告 / 实验追踪——每一段都是 trade-learn 的一等公民，不需要在框架之外另接 pyfolio / alphalens / empyrical / quantstats。

## 它适合谁

- **量化研究者**：希望同一份代码同时跑研究、回测、报告，且数值和原生 empyrical / pyfolio / alphalens 严格一致
- **策略开发者**：想要 backtrader 风格的事件驱动 API，但需要 Rust 内核级别的性能和可复现性
- **机器学习实验者**：需要把 sklearn / 因果选择 / 因子组合 / 回测和 MLflow 串起来，不重复造轮子
- **A 股 / 海外双市场用户**：需要 TDX（通达信口径）和 TradingView 双口径指标在同一框架下显式可选

## 先看什么

| 你的目标 | 推荐阅读 |
|---|---|
| 30 行代码跑通第一个回测 | [快速开始](quickstart.md) |
| 写一个轻量策略 | [Lite 指南](guides/lite.md) |
| 迁移 backtrader 风格策略 | [Engine 指南](guides/engine.md) |
| 理解整体架构 | [架构与边界](concepts/architecture.md) |
| 做因子、机器学习、组合研究 | [研究流水线](guides/research.md) |
| 查看性能与对齐结果 | [性能基线](benchmarks.md) |
| 想理解撮合 / 记账 / 一致性内核 | [设计笔记](internals/contracts.md) |
| 查完整 API | [API 参考](api/reference.md) |

## 核心能力

- **Lite 快速入口**——短语法，适合快速验证、教学、1.x 风格迁移和多资产目标权重
- **Engine 专业入口**——backtrader 风格 `Cerebro / Strategy / Analyzer / Sizer / Signal`，适合复杂事件驱动策略
- **Rust 回测内核**——单事件 `EventRunner`，single / multi-data runner 自动选择，回测和实盘共用同一套语义
- **双市场指标生态**——A 股偏 TDX / MyTT 口径（`tl.tdx` / `bt.tdx`），海外偏 TradingView（`tl.tv` / `bt.tv`）+ pandas-ta-classic / TA-Lib（`tl.pta` / `tl.talib`），用户显式选择
- **研究流水线**——`FeatureSet` / `Pipeline` / `CausalSelector` / `ResearchRun` / `Allocator` 把训练 / 测试 / 预处理 / 评分 / 权重 / 回测连成一条链
- **Optuna / MLflow / JupyterLab / MCP**——参数搜索、实验追踪、交互研究、LLM 工具集成

## 入口选择

trade-learn 提供两个用户入口，**底层共享同一套 Rust 回测 runtime，结果数值一致**：

| 入口 | 适合场景 | 运行结果 |
|---|---|---|
| `tradelearn.lite` | 快速策略、多资产目标权重、轻量研究、1.x 风格迁移 | `LiteStats`，字段与 Engine `Stats` 对齐 |
| `tradelearn.engine` | backtrader 风格专业策略、Analyzer、Sizer、Signal、未来 paper / live 模式 | `Strategy.stats` |
| `tradelearn.research` | 特征、预处理、选股、权重、研究记录 | `ResearchResult` |
| `tradelearn.factor` | alphalens 风格单因子 / 多因子分析 | `FactorAnalyzer` 报告 |
| `tradelearn.report` | pyfolio 风格收益、回撤、交易报告 | HTML / CSV / XLSX artifacts |

> `tradelearn.backtest.*` 和 `tradelearn.core.*` 是两个 facade 共享的实现层与中性契约——**不是公开用户 API**。请勿直接 import。Paper / live 适配从 `tradelearn.engine` 一侧扩展，不会再复制一套 backtest runtime。

## 一致性承诺

trade-learn 把"对照基线"当作工程纪律：

- **metrics**（sharpe / max_dd / sortino / ...）对 empyrical：`rtol=1e-10`
- **`tl.pta` / `bt.pta`** 对 pandas-ta-classic：`rtol=1e-10`
- **`tl.tdx` / `bt.tdx`** 对 MyTT：`rtol=1e-10`
- **`tl.tv` / `bt.tv`** 对 pyneCore / TradingView 截图：`rtol=1e-6`
- **回测 trades**（决策层）对 backtrader oracle：**0 差异**（时间 / 方向 / size）
- **回测 equity** 对 backtrader oracle：`rtol=1e-6`
- **回测 summary** 对 backtrader oracle：`rtol=1e-4`，差异必须可解释、登记在案

详见 [与外部库的语义一致性](internals/consistency.md)。
