# Overview

本页由 `scripts/generate_api_reference.py` 自动生成。

把这里当作 Tradelearn 的 API 地图：
- **Guide**：侧重“怎么用”，提供场景化示例和逻辑说明。
- **Reference**：侧重“是什么”，提供精确的类/函数签名、参数定义和 Docstrings。

## 快速链接

| 目标 | 阅读 |
|---|---|
| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API 签名](../guides/lite-api.md) |
| 编写 Backtrader 风格高级策略 | [Engine API 签名](../guides/engine-api.md) |
| 理解两种入口的设计差异 | [策略编写指南](../guides/strategy.md) |

## 公开模块矩阵

| 模块 | 用途 | 常用入口 | 完整 Reference |
|---|---|---|---|
| `tradelearn.lite` | Lightweight Tradelearn 1.x style API. | `Backtest`, `Strategy`, `ta`, `pta`, `talib`, `tdx`, `tv` | [Lite](reference/lite.md) |
| `tradelearn.engine` | Backtrader-style advanced event-driven API. | `Cerebro`, `OptReturn`, `DataFeed`, `CommInfoBase`, `ExecutedInfo`, `LineSeries`, `Order`, `Params`, `Position`, `Sizer`, ... (+40) | [Engine](reference/engine.md) |
| `tradelearn.data` | OHLCV data providers, caching, and resampling utilities. | `BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TradingViewProvider` | [Data](reference/data.md) |
| `tradelearn.indicators` | Technical indicator facade for pandas-ta-classic, TA-Lib, TDX, and TradingView styles. | `FunctionIndicator`, `Indicator`, `adx`, `atr`, `bbands`, `ema`, `macd`, `rsi`, `pta`, `talib`, ... (+4) | [Indicators](reference/indicators.md) |
| `tradelearn.metrics` | Return, risk, drawdown, trade, and factor evaluation metrics. | `alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, ... (+23) | [Metrics](reference/metrics.md) |
| `tradelearn.factor` | Alpha formulas and factor analysis tools. | `FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns` | [Factor](reference/factor.md) |
| `tradelearn.report` | HTML, Excel, and research report export utilities. | `Reporter`, `ReportContext`, `ReportSection` | [Report](reference/report.md) |
| `tradelearn.research` | Research workflow tracking, preprocessing, and portfolio construction tools. | `Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, ... (+7) | [Research](reference/research.md) |
| `tradelearn.ml` | Machine learning, model registry, and feature selection utilities. | `CausalSelector`, `AutoML`, `ModelLoader`, `ModelRegistry`, `model_uri` | [ML](reference/ml.md) |
