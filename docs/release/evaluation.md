# Evaluation

下列性能表来自存量 baseline 文件，不是本轮新冻结/新产物的性能验收；本轮产物按文末的源码闭包和实际安装/运行门禁记录。

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
| vnpy | 交易系统与实盘网关生态 | 当前源码聚焦本地研究与回测，执行走独立 NA 宿主，可选适配器单独验收 |
| backtrader | 成熟的事件驱动回测 API | Engine 按 Backtrader 公开契约迁移，Lite 保研究便利；具体行为以合格回归为准 |
| nautilus | 高性能事件驱动交易架构 | trade-learn 采用 Rust 事件型撮合核,但保留轻量 Python SDK 使用体验 |

### 2.0 定位

本小节保留早期规划标题，不代表已发行 2.0。现行源码和已公开版本分别记录。

- Engine/Lite 共用运行事实，公开入口与便利能力按各自契约提供
- Rust 事件型撮合核负责回测一致性与性能
- 实盘使用独立 `tradelearn_runtime.NautilusStore` 和官方 NA 宿主；适配器资格单独验收，不改变现役策略

## 候选产物的源码闭包验收

历史性能表与旧源码测试收据不能证明新候选已经合格。发行前保存全部构建输入的摘要，核构建前后未变；同时取得各域本轮的 `SOURCE-READY.json`（`files` 为相对源码路径到 SHA256 的映射）。用实际编译的 wheel 和 sdist 核声明文件：

```bash
python scripts/check_release_closure.py --root . \
  --wheel /path/to/actual.whl --sdist /path/to/actual.tar.gz \
  --manifest /path/to/SOURCE-READY.json --out /path/to/closure.json
```

多个冻结清单使用重复的 `--manifest`。工具逐项核当前源码、sdist 中全部声明文件、wheel 中声明的包文件、完整 RECORD 和私有构建路径；摘要冲突、漏装、旧源码或损坏产物均拒绝，失败会覆盖原成功收据。测试中的小型合成归档只验证拒绝机制，不能充当实际发行物证据。

成功收据仅证明字节闭包，`runtime_verified` 保持 false。之后仍须用新环境安装对应 profile，执行真实原生生命周期、研究/报告/模型恢复等适用验收，并以实际目标平台运行收据限定支持范围。报告中必须记录路径、摘要、解释器、安装命令和失败；不得仅以各 owner 停止改动、旧版本测试通过或 metadata 解析成功宣称最终验收完成。源码安装组、公开发行版本和本地候选分别标明。
