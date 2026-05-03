# API 参考

本页由 `scripts/generate_api_reference.py` 自动生成。

把这里当作 API 地图:需要示例和调用流程时看指南;需要完整类、函数、参数签名时看模块参考。

## 先看这里

| 目标 | 阅读 |
|---|---|
| 编写 Tradelearn 1.x 风格轻量策略 | [Lite API 签名](../guides/lite-api.md) |
| 编写 Backtrader 风格事件策略、Analyzer、Observer、Sizer | [Engine API 签名](../guides/engine-api.md) |
| 从零编写策略并理解两种入口差异 | [策略编写指南](../guides/strategy.md) |
| 查询精确类/函数签名和完整 docstring | 下方模块参考链接 |

## 公开模块

| 模块 | 用途 | 常用入口 | 完整参考 |
|---|---|---|---|
| `tradelearn.lite` | Tradelearn 1.x 风格轻量 API。 | `Backtest`, `Strategy`, `ta`, `pta`, `talib`, `tdx`, `tv` | [Lite](reference/lite.md) |
| `tradelearn.engine` | Backtrader 风格高级事件驱动 API。 | `Cerebro`, `OptReturn`, `DataFeed`, `CommInfoBase`, `ExecutedInfo`, `LineSeries`, `Order`, `Params`, `Position`, `Sizer`, ... (+40) | [Engine](reference/engine.md) |
| `tradelearn.data` | K 线数据、provider、缓存与重采样工具。 | `BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TradingViewProvider` | [数据](reference/data.md) |
| `tradelearn.indicators` | 技术指标 facade,包含 pandas-ta-classic / TDX / TradingView 生态入口。 | `FunctionIndicator`, `Indicator`, `adx`, `atr`, `bbands`, `ema`, `macd`, `rsi`, `pta`, `talib`, ... (+4) | [指标](reference/indicators.md) |
| `tradelearn.metrics` | 收益、风险、回撤与因子评价指标。 | `alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, ... (+23) | [收益风险指标](reference/metrics.md) |
| `tradelearn.factor` | Alpha 公式与因子分析工具。 | `FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns` | [因子](reference/factor.md) |
| `tradelearn.report` | HTML、Excel 与研究报告导出。 | `Reporter`, `ReportContext`, `ReportSection` | [报告](reference/report.md) |
| `tradelearn.research` | 指数增强研发流程记录、预处理与组合构建工具。 | `Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, ... (+7) | [研究](reference/research.md) |
| `tradelearn.ml` | 机器学习、模型注册与特征筛选。 | `CausalSelector`, `AutoML`, `ModelLoader`, `ModelRegistry`, `model_uri` | [机器学习](reference/ml.md) |

## 按模块列出公开符号

### `tradelearn.lite`

`Backtest`, `Strategy`, `ta`, `pta`, `talib`, `tdx`, `tv`

### `tradelearn.engine`

`Cerebro`, `OptReturn`, `DataFeed`, `CommInfoBase`, `ExecutedInfo`, `LineSeries`, `Order`, `Params`, `Position`, `Sizer`, `Strategy`, `Trade`, `Indicator`, `TimeFrame`, `FixedSize`, `PercentSizer`, `AllInSizer`, `Analyzer`, `feeds`, `analyzers`, `az`, `observers`, `sizers`, `GridSearchResult`, `grid_search`, `Observer`, `obs`, `CommissionInfo`, `pta`, `talib`, `tdx`, `tv`, `Signal`, `SignalStrategy`, `SIGNAL_NONE`, `SIGNAL_LONGSHORT`, `SIGNAL_LONG`, `SIGNAL_LONG_INV`, `SIGNAL_LONG_ANY`, `SIGNAL_SHORT`, `SIGNAL_SHORT_INV`, `SIGNAL_SHORT_ANY`, `SIGNAL_LONGEXIT`, `SIGNAL_LONGEXIT_INV`, `SIGNAL_LONGEXIT_ANY`, `SIGNAL_SHORTEXIT`, `SIGNAL_SHORTEXIT_INV`, `SIGNAL_SHORTEXIT_ANY`, `num2date`, `date2num`

### `tradelearn.data`

`BarsCache`, `CacheEntry`, `CacheExpiredError`, `CacheMissError`, `DataExplorer`, `DataProvider`, `DuckDBBarsBackend`, `TdxProvider`, `TradingViewProvider`

### `tradelearn.indicators`

`FunctionIndicator`, `Indicator`, `adx`, `atr`, `bbands`, `ema`, `macd`, `rsi`, `pta`, `talib`, `sma`, `tdx`, `tv`, `vwap`

### `tradelearn.metrics`

`alpha`, `annual_return`, `autocorrelation`, `avg_loss`, `avg_win`, `beta`, `calmar`, `cum_returns`, `cvar`, `downside_risk`, `drawdown_series`, `excess_returns`, `expectancy`, `factor_returns`, `ic`, `ic_ir`, `information_ratio`, `log_to_simple`, `max_consecutive_losses`, `max_consecutive_wins`, `max_drawdown`, `omega`, `profit_factor`, `quantile_returns`, `rank_ic`, `sharpe`, `simple_returns`, `sortino`, `tail_ratio`, `turnover`, `var`, `volatility`, `win_rate`

### `tradelearn.factor`

`FactorAnalyzer`, `MultiFactorAnalyzer`, `MultiPeriodFactorAnalyzer`, `FactorRiskModel`, `PerformanceAttribution`, `alpha101`, `alpha191`, `clean_factor_and_forward_returns`

### `tradelearn.report`

`Reporter`, `ReportContext`, `ReportSection`

### `tradelearn.research`

`Pipeline`, `FeatureBuilder`, `FeatureSet`, `ModelScorer`, `ResearchResult`, `ResearchRun`, `ResearchStep`, `Transformer`, `current_run`, `derive`, `explore`, `portfolio`, `preprocess`, `split`, `split_bars`, `time_split`, `tracked`

### `tradelearn.ml`

`CausalSelector`, `AutoML`, `ModelLoader`, `ModelRegistry`, `model_uri`


## 自动生成页面

- [Lite API 签名](../guides/lite-api.md)
- [Engine API 签名](../guides/engine-api.md)
- [策略编写指南](../guides/strategy.md)
- [`tradelearn.lite`](reference/lite.md)
- [`tradelearn.engine`](reference/engine.md)
- [`tradelearn.data`](reference/data.md)
- [`tradelearn.indicators`](reference/indicators.md)
- [`tradelearn.metrics`](reference/metrics.md)
- [`tradelearn.factor`](reference/factor.md)
- [`tradelearn.report`](reference/report.md)
- [`tradelearn.research`](reference/research.md)
- [`tradelearn.ml`](reference/ml.md)
