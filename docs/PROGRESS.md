# Current Progress

最后更新:2026-04-29

这份文档只保留当前项目状态、阶段摘要和下一步。完整历史流水已归档到
[`docs/archive/progress-2026-04.md`](./archive/progress-2026-04.md)。

判断以代码、已提交记录、CI、runner 输出和金标测试结果为准,不按历史对话口头状态判断。

## 执行规则

1. 先从最早阶段扫描未完成项,再推进当前任务。
2. `core` 只保留通用事件驱动运行核心,不得反向依赖 `tradelearn.compat.*`。
3. Backtrader 专属 API 放在 `tradelearn/compat/backtrader`,backtesting.py 专属 API 放在 `tradelearn/compat/backtesting`。
4. 指标公式由 pandas-ta-classic、TDX、TradingView 集成维护;core 只放通用缓存/代理机制。
5. QMT 等实盘适配暂不提交具体 broker 文件,只保留通用扩展接口。
6. 勾选状态必须对应已提交代码和验证结果;跳过/降级必须在对应 spec 或 migration 文档登记。

## 当前状态

| | |
|---|---|
| 当前阶段 | 阶段 9 发版 + 阶段 10 前置优化并行 |
| 总路线图完成度 | 约 80% |
| 已完成主线 | 工程地基、metrics、factor/report、Rust 撮合核、CLI、ML 能力、compat.backtrader、compat.backtesting 对齐与性能前置优化 |
| 当前性能基线 | `compare_backtesting.py` 当前实测约 1.5x,其中 BTCUSDT 约 1.49x-1.56x、ETHUSDT 约 1.46x-1.49x;`benchmark_bt.py smart --warmup 1 --repeat 3 --min-speedup 1.2` 8/8 EXACT,约 2.5x-7.5x |
| 下一里程碑 | Stage 9 Week 2: wheel 含 Rust 二进制 -> PyPI / GitHub Release + NOTICE 最终审查 |
| Stage 10 状态 | Rust callback loop、订单缓冲、fill 增量同步、`_pre_next`、Rust primary-clock 多数据游标计划、共享 bar buffer、BrokerEventPump、lazy stats 与 backtesting facade 热路径修补已完成;后续优化坚持单一 core runner,不新增 compat 专属 runner 或 slim mode |

## 阶段总览

| 阶段 | 主题 | 状态 | 说明 |
|---|---|---|---|
| 0 | 地基 + 金标基线 | ✅ 完成 | TV subset oracle/parity 链路闭合;TDX live 数据因海外访问限制永久放弃为 golden 主线阻塞 |
| 1 | metrics 融合 | ✅ 完成 | returns/risk/factor/trade API 与 consistency 测试已闭合 |
| 2 | factor/report + 依赖替换 | ✅ 完成 | FactorAnalyzer、Reporter、Excel/HTML、Alpha101/191、设计笔记冻结已完成 |
| 冷冻期 | Rust/PyO3 脚手架 + PoC | ✅ 完成 | Cargo workspace、PyO3、maturin、pyneCore PoC 已完成 |
| 3 | Rust 事件撮合核 | ✅ 完成 | exact/smart 两套 K 线撮合、Portfolio、Stats、Analyzer、多数据 primary-clock 与 TV subset full golden 对照已闭合 |
| 4 | 数据缓存 + Analyzer + CLI | ✅ 完成 | BarsCache、MLflowAnalyzer、grid_search、Typer CLI、Config 系统已落地 |
| 5 | JupyterLab 环境 | 🟡 等待 Stage 7 | lab dry-run、starter notebooks、doctor 已就绪;完整 MCP/Chat 面板依赖 Stage 7 |
| 6 | ML 能力 | ✅ 完成 | MLStrategy、FeatureStore、ModelRegistry、CausalSelector、sklearn GBM + Alpha101 示例已闭合 |
| 7 | MCP Server | ↪️ deferred | 完整 MCP Server 与 Jupyter AI persona 暂缓 |
| 8 | compat.backtrader | ✅ 完成 | Cerebro/Strategy/feeds/indicators/notify 与 10 个迁移策略已闭合 |
| 9 | 文档 + 发版 | 🟡 进行中 | 文档、release golden gate、examples 策略目录整理已完成;剩余 wheel/PyPI/GitHub Release |
| 10 | QMT 实盘 + Rust BarRunner | 🟡 进行中 | QMT 具体文件暂不提交;事件驱动实盘兼容接口保留 |

## 当前验证入口

```bash
uv run pytest tests/unit/backtest/test_compat_runner_scripts.py -q
uv run pytest tests/unit/examples/test_examples_layout.py -q
uv run python benchmarks/runners/compare_backtesting.py
uv run python benchmarks/runners/benchmark_bt.py smart --warmup 1 --repeat 3 --min-speedup 1.2
```

## 最近关键提交

- `237bec3` core line primitives 下沉,core 不再依赖 compat。
- `04e4634` 补项目结构文档并忽略本地 artifacts。
- `59c8164` 保证 `examples/` 只保留策略文件,runner/data 移出。
- `a7835cc` 在不新增 runner 的前提下压缩 backtesting.py facade 每 bar 热路径。
- `aea60ad` 引入 lazy stats materialization,保留完整 artifacts 能力但默认不急切构建 pandas 产物。
- 本轮整理:临时脚本/IDE 配置移出 git,历史进度归档,文档门禁改用 `PROJECT.md` 作为愿景/路线入口。

## 当前待办

1. Stage 9: 构建并验证含 Rust 二进制的 wheel。
2. Stage 9: PyPI / GitHub Release / NOTICE 最终审查。
3. Stage 10: 若继续优化,优先 profile 证明后的 proxy/submit 微优化和正式 benchmark gate;暂不推进独立 fast runner、core slim mode、批 callback 或策略下沉。
4. 清理类任务: 后续可拆分超大测试文件 `test_rust_exact_matching.py`、`test_strategy_api.py`、`test_alpha_metadata.py`。
