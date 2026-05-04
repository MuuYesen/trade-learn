# 对齐与性能基线

trade-learn 的正确性先看 Engine。`tradelearn.engine` 以 Backtrader 为 oracle，`benchmarks/runners/benchmark_bt.py` 会对代表性策略做最终权益、交易明细和 PnL 对齐；当前主线要求 Engine 与 Backtrader 保持 `EXACT`。

Lite 不是 Backtrader facade，它是更薄的策略语法层。Lite 与 Engine 共用 `tradelearn.backtest` runtime 和 Rust 撮合内核，因此 Lite 的验收重点是：API 能正确接入同一 runtime，返回统计字段与 Engine 一致，最终权益 / 成交数 / 平仓交易数与同一策略语义保持一致。

## 大样本性能

本机基线只看两个问题：结果是否对齐、吞吐是否明显快于 Backtrader。

### 55 万 bar 单标的 SMA

验证入口：[`benchmarks/runners/benchmark_throughput.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_throughput.py)。该 runner 使用文件内定义的 synthetic SMA 策略，避免示例文件变动影响吞吐基线。

| 引擎 | 耗时 | bars/s | 加速比 | Final Value | Fills | Closed Trades |
|---|---:|---:|---:|---:|---:|---:|
| Tradelearn Lite | 1.3253s | 414,990 | 27.9x | 118,399.33 | 10,299 | 5,149 |
| Tradelearn Engine | 3.3767s | 162,883 | 11.0x | 118,399.33 | 10,299 | 5,149 |
| Backtrader | 37.0270s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 |

### 1000 标的、约 20 年、月频 top-50 目标权重

总计 5,040,000 根 data bars。

验证入口：[`benchmarks/runners/benchmark_target_weight_parity.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_target_weight_parity.py)。该 runner 使用文件内定义的 synthetic target-weight 策略，专门隔离多标的调仓、订单生命周期和吞吐表现。

| 引擎 | 耗时 | bars/s | 加速比 | Final Value | Completed Orders | Target Intents | Targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tradelearn Lite | 2.407s | 2,094,237 | 119.1x | 4,199,638.26 | 23,249 | 23,249 | 239 |
| Tradelearn Engine | 4.112s | 1,225,594 | 69.7x | 4,199,638.26 | 23,249 | 23,249 | 239 |
| Backtrader | 286.538s | 17,589 | 1.0x | 4,199,638.26 | 23,249 | 23,249 | 239 |

结论：Engine 与 Backtrader 在最终权益、完成订单数、目标意图数和 rebalance 次数上全部 `EXACT`；Engine 约为 Backtrader 的 **69.7x**，Lite 约为 **119.1x**。这些数字不作为跨机器的绝对性能承诺；正式质量门禁仍然是 Engine 与 Backtrader 的订单生命周期和交易明细保持 `EXACT`。

### Research pipeline 分段耗时

验证入口：[`benchmarks/runners/benchmark_research_pipeline.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_research_pipeline.py)。该 runner 只看 Stage 12 投研主路径的分段耗时，不和 Backtrader 对比：`panel -> factor -> weights -> backtest -> report -> MLflow artifacts`。

本机轻量样本：50 标的、240 bars、top-10 组合，合计 12,000 data bars。

| 分段 | 耗时 | 占比 | 含义 |
|---|---:|---:|---|
| `panel` | 0.0292s | 4.0% | 合成/准备 MultiIndex OHLCV panel，并切出 test bars |
| `factor` | 0.0330s | 4.6% | `FeatureSet` 生成因子与标签，`research.Pipeline` fit/transform |
| `weights` | 0.1029s | 14.2% | `Allocator(TopK + EqualWeight + Constraints)` 生成多日期目标权重 |
| `backtest` | 0.0247s | 3.4% | Lite 通过共享 backtest runtime 执行 `target_weights()` |
| `report` | 0.4422s | 61.0% | Reporter 写 `report.html` |
| `mlflow_artifacts` | 0.0926s | 12.8% | 生成 MLflow 可上传的 `artifacts.xlsx` 与拆分 CSV |
| **total** | **0.7246s** | **100.0%** | Final Value 1,011,868.22，Return 1.19%，Trades 9，Weight dates 64 |

## Examples 对齐审计

`examples/` 里的策略用于检查用户可读策略是否还能保持 Backtrader 语义对齐，不混入大样本吞吐统计。

### 单标的 Engine 策略

验证入口：[`benchmark_bt.py`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_bt.py)。每个策略都用 Tradelearn Engine 和 Backtrader 各跑一遍，比较最终权益、平仓交易数、累计 PnL 和扣费后 PnL。

| 策略 | Final Value TL / BT | Closed Trades TL / BT | Closed PnL TL / BT | PnLComm TL / BT | 时间 TL / BT | 状态 |
|---|---:|---:|---:|---:|---:|---|
| [`QuickstartSmaCross`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/01_quickstart.py) | 100026.14 / 100026.14 | 16 / 16 | 26.14 / 26.14 | 26.14 / 26.14 | 15.3ms / 16.2ms | EXACT |
| [`SmaCross`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/02_sma_cross.py) | 99630.56 / 99630.56 | 3 / 3 | -247.72 / -247.72 | -247.72 / -247.72 | 11.1ms / 15.0ms | EXACT |
| [`MigratedSmaCross`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/05_migration.py) | 99997.70 / 99997.70 | 21 / 21 | -2.30 / -2.30 | -2.30 / -2.30 | 9.8ms / 17.1ms | EXACT |
| [`Turtle`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/06_turtle.py) | 99995.64 / 99995.64 | 8 / 8 | -4.36 / -4.36 | -4.36 / -4.36 | 15.2ms / 26.4ms | EXACT |
| [`EnhancedRSI`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/07_rsi_enhanced.py) | 97875.79 / 97875.79 | 6 / 6 | -2124.21 / -2124.21 | -2124.21 / -2124.21 | 12.3ms / 21.8ms | EXACT |
| [`BetterMA`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/08_better_ma.py) | 100000.00 / 100000.00 | 0 / 0 | 0.00 / 0.00 | 0.00 / 0.00 | 7.5ms / 15.1ms | EXACT |
| [`MacdTharp`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/09_macd_settings.py) | 99998.98 / 99998.98 | 2 / 2 | -1.02 / -1.02 | -1.02 / -1.02 | 20.2ms / 17.0ms | EXACT |
| [`OrderExecutionStrategy`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/10_order_execution.py) | 99994.05 / 99994.05 | 13 / 13 | -5.95 / -5.95 | -5.95 / -5.95 | 20.1ms / 16.5ms | EXACT |

### 多资产组合 Engine 策略

验证入口：[`benchmark_bt.py --include-portfolio`](https://github.com/MuuYesen/trade-learn/blob/master/benchmarks/runners/benchmark_bt.py)。这些策略覆盖 `order_target_percent`、资产类别权重、趋势过滤、反波动权重等组合调仓语义。

| 策略 | Final Value TL / BT | Orders TL / BT | 时间 TL / BT | 加速比 | 状态 |
|---|---:|---:|---:|---:|---|
| [`TargetPercentPortfolioStrategy`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/11_target_percent_portfolio.py) | 104447.50 / 104447.50 | 14 / 14 | 9.4ms / 26.8ms | 2.8x | EXACT |
| [`AssetClassTargetPortfolioStrategy`](https://github.com/MuuYesen/trade-learn/blob/master/examples/engine/12_asset_class_portfolios.py) | 104003.95 / 104003.95 | 21 / 21 | 8.7ms / 27.8ms | 3.2x | EXACT |
| `UniformAssetClassPortfolioStrategy` | 104155.45 / 104155.45 | 22 / 22 | 9.0ms / 27.2ms | 3.0x | EXACT |
| `TrendFilteredPortfolioStrategy` | 103430.20 / 103430.20 | 21 / 21 | 8.6ms / 27.7ms | 3.2x | EXACT |
| `InverseVolatilityPortfolioStrategy` | 104410.00 / 104410.00 | 9 / 9 | 8.8ms / 27.4ms | 3.1x | EXACT |

### Lite / Engine research workflow

Workflow 示例不以 Backtrader 为 oracle，而是验证投研结果能进入同一套 `Stats` / report / MLflow artifacts。

| 范围 | 完整示例 |
|---|---|
| Lite 投研 + 回测 + report + MLflow | [`examples/research/index_enhance_lite_pipeline.py`](https://github.com/MuuYesen/trade-learn/blob/master/examples/research/index_enhance_lite_pipeline.py) |
| Engine 投研 + 回测 + report + MLflow | [`examples/research/index_enhance_engine_pipeline.py`](https://github.com/MuuYesen/trade-learn/blob/master/examples/research/index_enhance_engine_pipeline.py) |

## 复现命令

```bash
uv run python benchmarks/runners/benchmark_bt.py smart --repeat 1 --warmup 0

uv run python benchmarks/runners/benchmark_bt.py smart --repeat 1 --warmup 0 --include-portfolio

uv run python benchmarks/runners/benchmark_throughput.py --bars 550000 --repeat 1 --warmup 0

uv run python benchmarks/runners/benchmark_target_weight_parity.py \
  --symbols 1000 --bars 5040 --holdings 50 --rebalance-every 21

uv run python benchmarks/runners/benchmark_research_pipeline.py \
  --symbols 50 --bars 240 --holdings 10
```
