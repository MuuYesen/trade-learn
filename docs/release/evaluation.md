# Evaluation

## 性能 Benchmark

| 场景 | 目标 | 实测 | 状态 |
|---|---:|---:|---|
| 单品种 10 年 | 50ms | 3.173ms | pass |
| 500 股组合 | 5s | 312.569ms | pass |

### 一致性口径

- trades 0 差异
- PnL rtol=1e-4
- 数据来自 `benchmarks/baseline.json` 的 `stage3_backtest` 记录

### 复现命令

`uv run python scripts/check_stage3_benchmark.py --single-bars 2520 --max-single-ms 50 --portfolio-bars 2520 --portfolio-symbols 500 --max-portfolio-ms 5000 --json`

## 竞品对比

vs qlib / vnpy / backtrader / nautilus

trade-learn 是 Python 量化研究框架,让传统策略与 ML 策略共用 API。

| 项目 | 侧重点 | trade-learn 差异 |
|---|---|---|
| qlib | 研究平台偏重数据、模型与实验体系 | trade-learn 是本地优先的 Python 量化研究框架,传统策略与 ML 策略共用 API |
| vnpy | 交易系统与实盘网关生态 | trade-learn 1.0 聚焦研究与回测 SDK,1.1 通过 QMT 补齐研究到实盘路径 |
| backtrader | 成熟的事件驱动回测 API | trade-learn 提供 compat.backtrader 迁移层,同时接入 模型组件与现代报告体系 |
| nautilus | 高性能事件驱动交易架构 | trade-learn 采用 Rust 事件型撮合核,但保留轻量 Python SDK 使用体验 |

### 1.0 定位

- compat.backtrader 承接存量策略
- Rust 事件型撮合核负责回测一致性与性能
- QMT 实盘对接进入 1.1,不拖慢 1.0 发版
