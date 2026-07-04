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
| `tradelearn.lite` | Lightweight Tradelearn 1.x style API. | `Backtest`, `Strategy`, `TargetOrderConstraints` | [Lite](reference/lite.md) |
| `tradelearn.engine` | Backtrader-style advanced event-driven API. | `Cerebro`, `OptReturn`, `DataFeed`, `CommInfoBase`, `ExecutedInfo`, `LineSeries`, `Order`, `Params`, `Position`, `Sizer`, ... (+43) | [Engine](reference/engine.md) |
| `tradelearn.data` | OHLCV data providers, caching, and resampling utilities. | `BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TradingViewProvider` | [Data](reference/data.md) |
| `tradelearn.indicators` | Technical indicator facade for core indicators and optional pandas-ta-classic, TA-Lib, TDX, and TradingView styles. | `FunctionIndicator`, `Indicator`, `adx`, `atr`, `bbands`, `ema`, `macd`, `rsi`, `sma`, `vwap` | [Indicators](reference/indicators.md) |
| `tradelearn.metrics` | Return, risk, drawdown, trade, and factor evaluation metrics. | `alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, ... (+24) | [Metrics](reference/metrics.md) |
| `tradelearn.factor` | Alpha formulas and factor analysis tools. | `FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns` | [Factor](reference/factor.md) |
| `tradelearn.report` | HTML, Excel, and research report export utilities. | `Reporter`, `ReportContext`, `ReportSection` | [Report](reference/report.md) |
| `tradelearn.research` | Research workflow tracking, preprocessing, and portfolio construction tools. | `Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, ... (+7) | [Research](reference/research.md) |
| `tradelearn.ml` | Machine learning, model registry, and feature selection utilities. | `CausalSelector`, `AutoML`, `ModelLoader`, `ModelRegistry`, `model_uri` | [ML](reference/ml.md) |

## 按模块列出公开符号

### [`tradelearn.lite`](reference/lite.md)

`Backtest`, `Strategy`, `TargetOrderConstraints`

### [`tradelearn.engine`](reference/engine.md)

`Cerebro`, `OptReturn`, `DataFeed`, `CommInfoBase`, `ExecutedInfo`, `LineSeries`, `Order`, `Params`, `Position`, `Sizer`, `Strategy`, `Trade`, `Indicator`, `TimeFrame`, `FixedSize`, `PercentSizer`, `AllInSizer`, `Analyzer`, `feeds`, `analyzers`, `az`, `observers`, `sizers`, `GridSearchResult`, `grid_search`, `Observer`, `obs`, `CommissionInfo`, `BarRangeSlippage`, `CNAStockCommission`, `FixedCommission`, `FixedSlippage`, `PercentCommission`, `PercentSlippage`, `TargetOrderConstraints`, `Signal`, `SignalStrategy`, `SIGNAL_NONE`, `SIGNAL_LONGSHORT`, `SIGNAL_LONG`, ... (+13)

### [`tradelearn.data`](reference/data.md)

`BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TradingViewProvider`

### [`tradelearn.indicators`](reference/indicators.md)

`FunctionIndicator`, `Indicator`, `adx`, `atr`, `bbands`, `ema`, `macd`, `rsi`, `sma`, `vwap`

### [`tradelearn.metrics`](reference/metrics.md)

`alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, `drawdown_series`, `excess_returns`, `expectancy`, `factor_returns`, `ic`, `ic_ir`, `information_ratio`, `log_to_simple`, `max_consecutive_losses`, `max_consecutive_wins`, `max_drawdown`, `omega`, `profit_factor`, `quantile_turnover`, `quantile_returns`, `rank_ic`, `sharpe`, `simple_returns`, `sortino`, `tail_ratio`, `turnover`, `var`, `volatility`, `win_rate`

### [`tradelearn.factor`](reference/factor.md)

`FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns`

### [`tradelearn.report`](reference/report.md)

`Reporter`, `ReportContext`, `ReportSection`

### [`tradelearn.research`](reference/research.md)

`Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, `explore`, `portfolio`, `preprocess`, `split`, `split_bars`, `time_split`, `tracked`

### [`tradelearn.ml`](reference/ml.md)

`CausalSelector`, `AutoML`, `ModelLoader`, `ModelRegistry`, `model_uri`
