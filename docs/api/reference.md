# API 参考

本页由 `scripts/generate_api_reference.py` 自动生成。

## 先看这里

把这里当作 Tradelearn 的 API 地图：
- **Guide**：侧重“怎么用”，提供场景化示例和逻辑说明。
- **Reference**：侧重“是什么”，提供精确的类/函数签名、参数定义和 Docstrings。

## 快速链接

| 目标 | 阅读 |
|---|---|
| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API 签名](../guides/lite-api.md) |
| 编写 Backtrader 风格高级策略 | [Engine API 签名](../guides/engine-api.md) |
| 理解两种入口的设计差异 | [策略编写指南](../guides/strategy.md) |

## 公开模块

| 模块 | 用途 | 常用入口 | 完整 Reference |
|---|---|---|---|
| `tradelearn.lite` | Lightweight Tradelearn 1.x style API. | `Backtest`, `Strategy`, `Indicator`, `TargetOrderConstraints` | [Lite](reference/lite.md) |
| `tradelearn.engine` | Backtrader-style advanced event-driven API. | `Cerebro`, `OptReturn`, `CommInfoBase`, `LineSeries`, `Order`, `Position`, `Sizer`, `Timer`, `timer`, `SESSION_START`, ... (+34) | [Engine](reference/engine.md) |
| `tradelearn.data` | OHLCV data providers, caching, and resampling utilities. | `BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TushareProvider`, `TradingViewProvider`, ... (+2) | [Data](reference/data.md) |
| `tradelearn.indicators` | Qualified TA-Lib, TDX and TradingView indicators with stateful execution. | `DEFAULT_SOURCE`, `FunctionIndicator`, `Indicator`, `IndicatorState` | [Indicators](reference/indicators.md) |
| `tradelearn.metrics` | Return, risk, drawdown, trade, and factor evaluation metrics. | `cash_flow_return`, `alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, ... (+25) | [Metrics](reference/metrics.md) |
| `tradelearn.factor` | Alpha formulas and factor analysis tools. | `FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns` | [Factor](reference/factor.md) |
| `tradelearn.report` | HTML, Excel, and research report export utilities. | `Reporter`, `ReportContext`, `ReportSection` | [Report](reference/report.md) |
| `tradelearn.research` | Research workflow tracking, preprocessing, and portfolio construction tools. | `Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, ... (+13) | [Research](reference/research.md) |
| `tradelearn.integrations.search` | Explicit parameter grid execution outside the Engine and core runtime. | `grid_search`, `GridSearchResult` | [Search](reference/search.md) |
| `tradelearn.integrations.mlflow` | Explicit post-run recording of completed results; native Run ownership. | `log`, `LogResult` | [MLflow](reference/mlflow.md) |
| `tradelearn.engine.stores` | Lazy public Store namespace; Cerebro owns the single official Nautilus driver. | `NautilusStore` | [Stores](reference/stores.md) |
| `tradelearn.ml` | Machine learning, model registry, and feature selection utilities. | `CausalSelector`, `ModelLoader`, `ModelRegistry`, `model_uri`, `save_bundle`, `load_bundle`, `predict_bundle` | [ML](reference/ml.md) |

## 按模块列出公开符号

### [`tradelearn.lite`](reference/lite.md)

`Backtest`, `Strategy`, `Indicator`, `TargetOrderConstraints`

### [`tradelearn.engine`](reference/engine.md)

`Cerebro`, `OptReturn`, `CommInfoBase`, `LineSeries`, `Order`, `Position`, `Sizer`, `Timer`, `timer`, `SESSION_START`, `SESSION_END`, `Strategy`, `Trade`, `Indicator`, `TimeFrame`, `Analyzer`, `feeds`, `indicators`, `ind`, `analyzers`, `observers`, `sizers`, `stores`, `Observer`, `obs`, `CommissionInfo`, `Signal`, `SignalStrategy`, `SIGNAL_NONE`, `SIGNAL_LONGSHORT`, `SIGNAL_LONG`, `SIGNAL_LONG_INV`, `SIGNAL_LONG_ANY`, `SIGNAL_SHORT`, `SIGNAL_SHORT_INV`, `SIGNAL_SHORT_ANY`, `SIGNAL_LONGEXIT`, `SIGNAL_LONGEXIT_INV`, `SIGNAL_LONGEXIT_ANY`, `SIGNAL_SHORTEXIT`, ... (+4)

### [`tradelearn.data`](reference/data.md)

`BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TushareProvider`, `TradingViewProvider`, `Dataset`, `open_dataset`

### [`tradelearn.indicators`](reference/indicators.md)

`DEFAULT_SOURCE`, `FunctionIndicator`, `Indicator`, `IndicatorState`

### [`tradelearn.metrics`](reference/metrics.md)

`cash_flow_return`, `alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, `drawdown_series`, `excess_returns`, `expectancy`, `factor_returns`, `ic`, `ic_ir`, `information_ratio`, `log_to_simple`, `max_consecutive_losses`, `max_consecutive_wins`, `max_drawdown`, `omega`, `profit_factor`, `quantile_turnover`, `quantile_returns`, `rank_ic`, `sharpe`, `simple_returns`, `sortino`, `tail_ratio`, `turnover`, `var`, `volatility`, `win_rate`

### [`tradelearn.factor`](reference/factor.md)

`FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns`

### [`tradelearn.report`](reference/report.md)

`Reporter`, `ReportContext`, `ReportSection`

### [`tradelearn.research`](reference/research.md)

`Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, `explore`, `portfolio`, `preprocess`, `split`, `split_bars`, `time_split`, `tracked`, `Allocator`, `TopK`, `EqualWeight`, `Constraints`, `RiskfolioOptimizer`, `RiskfolioWeight`

### [`tradelearn.integrations.search`](reference/search.md)

`grid_search`, `GridSearchResult`

### [`tradelearn.integrations.mlflow`](reference/mlflow.md)

`log`, `LogResult`

### [`tradelearn.engine.stores`](reference/stores.md)

`NautilusStore`

### [`tradelearn.ml`](reference/ml.md)

`CausalSelector`, `ModelLoader`, `ModelRegistry`, `model_uri`, `save_bundle`, `load_bundle`, `predict_bundle`
