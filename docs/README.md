# trade-learn 技术手册

<p align="center">
  <img src="tradelearn-logo.png" alt="trade-learn logo" width="520" />
</p>

trade-learn 是一个面向量化研究、机器学习策略和事件驱动回测的 Python/Rust 框架：Python 负责表达策略、因子、模型和投研流程，Rust 负责撮合、订单推进、bar runner 和 portfolio 这类高频回测内核。

它的目标不是只跑一段回测，而是把一条完整策略研发链路接起来：

<p align="center">
  <img src="research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## 先看什么

| 目标 | 推荐阅读 |
|---|---|
| 快速写一个轻量策略 | [Lite Guide](guides/lite.md) |
| 迁移 Backtrader 风格策略 | [Engine Guide](guides/engine.md) |
| 理解整体架构 | [架构与边界](concepts/architecture.md) |
| 做因子、机器学习和组合研究 | [研究流水线](guides/research.md) |
| 查看性能与对齐结果 | [对齐与性能基线](benchmarks.md) |
| 查完整 API | [API Reference](api/reference.md) |

## 核心能力

- **Lite 快速入口**：短语法，适合快速验证、教学、1.x 风格迁移和多资产目标权重。
- **Engine 专业入口**：Backtrader 风格 `Cerebro / Strategy / Analyzer / Sizer / Signal`，适合复杂事件驱动策略。
- **Rust 回测内核**：single-data 和 multi-data runner 自动选择，用户仍然只写事件驱动策略。
- **双市场指标生态**：A 股偏 TDX/MyTT 口径，海外和通用研究偏 TradingView、TA-Lib、pandas-ta-classic。
- **研究流水线**：FeatureSet、Pipeline、CausalSelector、ResearchRun、Allocator 连接 train/test、预处理、评分、权重和回测。
- **Optuna / MLflow / JupyterLab / MCP**：参数搜索、实验追踪、交互研究和自动化工具集成。

## 入口选择

| 入口 | 适合场景 | 运行结果 |
|---|---|---|
| `tradelearn.lite` | 快速策略、多资产目标权重、轻量研究 | `LiteStats`，字段与 Engine `Stats` 对齐 |
| `tradelearn.engine` | Backtrader 风格专业策略、Analyzer、Sizer、Signal | `Strategy.stats` |
| `tradelearn.research` | 特征、预处理、选股、权重、研究记录 | `ResearchResult` |
| `tradelearn.factor` | alphalens 风格单因子 / 多因子分析 | `FactorAnalyzer` 报告 |
| `tradelearn.report` | pyfolio 风格收益、回撤、交易报告 | HTML / CSV / XLSX artifacts |
