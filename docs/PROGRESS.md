# Current Progress

最后更新:2026-04-29

这份文档只保留当前项目状态、阶段摘要和下一步。完整历史流水已归档到
[`docs/archive/progress-2026-04.md`](./archive/progress-2026-04.md)。

判断以代码、已提交记录、CI、runner 输出和金标测试结果为准,不按历史对话口头状态判断。

## 执行规则

1. 先从最早阶段扫描未完成项,再推进当前任务。
2. `core` 只保留通用事件驱动运行核心,不得反向依赖 `tradelearn.engine.* 或 tradelearn.lite.*`。
3. Backtrader 风格完整 API 放在 `tradelearn/engine`,入门 API 放在 `tradelearn/lite`；`compat` 迁移层已废弃。
4. 指标公式由 pandas-ta-classic、TDX、TradingView 集成维护;core 只放通用缓存/代理机制。
5. QMT 等实盘适配暂不提交具体 broker 文件,只保留通用扩展接口。
6. 勾选状态必须对应已提交代码和验证结果;跳过/降级必须在对应 spec 或 migration 文档登记。

## 当前状态

| | |
|---|---|
| 当前阶段 | 功能补齐优先;Stage 9 发版闭环暂缓 |
| 总路线图完成度 | 约 82% |
| 已完成主线 | 工程地基、metrics、factor/report、Rust 撮合核、CLI、ML 能力、`engine` 高级 API、`lite` 1.x 低级 API、共享 backtest runtime 与 Rust kernel |
| 当前性能基线 | `benchmark_bt.py` 作为 Backtrader EXACT 小样本审计保持 8/8 EXACT;`benchmark_throughput.py --bars 550000` 当前 55 万 bar 实测: Engine 约 180,364 bars/s,Lite 约 273,334 bars/s,Backtrader 约 16,369 bars/s;三者 Final Value / Fills / Closed Trades 口径一致 |
| 下一里程碑 | 功能优先:补齐 `engine` Backtrader 常用高级接口,并要求新增 broker/order 能力在 `engine` 与 `lite` 同步暴露;Stage 9 wheel/PyPI/GitHub Release 暂缓 |
| Stage 10 状态 | 主线性能优化已收口;Rust callback loop、订单缓冲、fill 增量同步、`_pre_next`、Rust primary-clock 多数据游标计划、共享 bar buffer、BrokerEventPump、lazy stats、Engine/Lite throughput 口径统一已完成;后续只接受 profile 证明的局部小修,不推进独立 fast runner、core slim mode、批 callback、DSL 或策略下沉 |

## 当前架构判定

- `tradelearn.engine` 是 Backtrader 风格高级 API,也是底层正确性的正式对齐入口。
- `tradelearn.lite` 是 Tradelearn 1.x 风格低级 API,只做语法适配与 smoke 验证,不再对齐 `backtesting.py`。
- `engine` 与 `lite` 都是 `tradelearn.backtest` runtime + Rust kernel 的语法翻译层;业务语义、撮合、订单推进、portfolio 必须一致。
- `core` 只放跨 backtest / paper / live 的中性契约,不得接收回测专属或 facade 专属逻辑。
- 指标公式不下沉 Rust,继续由 pandas-ta-classic、TDX、TradingView 等 Python 指标生态维护;Rust 只负责撮合、bar loop、订单推进、portfolio 等高性能回测内核。
- QMT / paper / live 暂不提交具体 broker 文件,只保留通用扩展接口和事件驱动边界。

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
| 8 | engine 高级 API | ✅ 完成 | Backtrader 风格 Cerebro/Strategy/feeds/indicators/notify 与迁移策略已闭合;继续补常用接口覆盖 |
| 9 | 文档 + 发版 | ⏸️ 暂缓 | 文档、release golden gate、examples 策略目录整理已完成;wheel/PyPI/GitHub Release 等发版闭环暂不优先 |
| 10 | QMT 实盘接口 + Rust BarRunner | 🟡 收口 | QMT 具体文件暂不提交;事件驱动实盘兼容接口保留;性能优化只做 profile-driven 小修 |
| 11 | 指数增强 / 全市场截面 | 📋 规划中 | DuckDB backend → 截面再平衡 API → Rust 多标的 portfolio → 因子归因 |

## 当前验证入口

```bash
uv run pytest tests/unit/lite tests/unit/examples/test_1x_strategy_examples.py -q
uv run pytest tests/unit/backtest/test_benchmark_runner_scripts.py -q
uv run pytest tests/unit/backtest/test_core_layering.py -q
uv run python benchmarks/runners/benchmark_bt.py
uv run python benchmarks/runners/benchmark_throughput.py --bars 550000 --repeat 1 --warmup 0
```

## 最近关键提交

- `87f6a20` 共享 feed 构造下沉到 backtest runtime,engine/lite 不互相依赖。
- `581391e` 拆分 Lite 策略支撑模块,`strategy.py` 保持薄 facade。
- `5123b6c` 清理 Lite facade 边界,移除 `Backtesting*` 命名残留并改用 broker 公共 storage API。
- `b32d24e` 记录 55 万 bar throughput 与 profiling 口径。
- `8f757be` 统一 throughput benchmark 语义,Engine/Lite/Backtrader 输出 Final Value / Fills / Closed Trades 一致,并修复 Lite `trade_on_close` 透传。

## 当前待办

1. 功能优先:补齐 `engine` 作为 Backtrader 风格高级 API 的常用接口覆盖;新增订单/broker 能力必须同时在 `engine` 与 `lite` 暴露。
2. Lite 继续保持 1.x 简洁语法,不扩成 `backtesting.py`;新增语法只作为 `engine` 语义的轻量翻译。
3. 正确性门禁:`benchmark_bt.py` 保持 8/8 EXACT;Lite smoke 与 1.x examples 持续通过;`benchmark_throughput.py` 保持 Final Value / Fills / Closed Trades 口径一致。
4. Stage 10 性能优化进入收口状态;后续只接受 profile 证明的局部小修,暂不推进独立 fast runner、core slim mode、批 callback、DSL 或策略下沉。
5. 清理类任务: 后续可拆分超大测试文件 `test_rust_exact_matching.py`、`test_strategy_api.py`、`test_alpha_metadata.py`。
6. Stage 9 发版闭环:wheel 含 Rust 二进制、PyPI / GitHub Release / NOTICE 最终审查暂缓,待功能补齐后恢复。

## Stage 11: 指数增强 / 全市场截面策略支持（规划中）

面向全市场股票（A 股 5000+）指数增强场景,分四阶段推进:

### 阶段 11.1 — DuckDB 数据 backend

- 在 `tradelearn/data/` 下新增 DuckDB 存储 backend,作为 `pd.read_csv` 的可选替代
- 支持列式裁剪（只拉策略需要的列）和日期范围懒加载
- 全市场日频 10 年（~1200 万行）能在秒级完成加载
- DuckDB 作为可选依赖（`pip install tradelearn[duckdb]`）

### 阶段 11.2 — 截面再平衡 API

- 在 `tradelearn/engine/` 新增截面策略生命周期:

  ```python
  class IndexEnhanceStrategy(Strategy):
      rebalance_freq = "monthly"
      def rebalance(self, dt, universe: pd.DataFrame) -> pd.Series:
          """输入: 当期全市场截面; 输出: 目标权重 Series"""
          ...
  ```

- 引擎按 `rebalance_freq` 自动触发,策略只关注选股/打分逻辑
- 不动现有 `next()` 逐 bar 驱动模型,新增并行 API

### 阶段 11.3 — Rust 多标的 portfolio 撮合

- Rust 引擎扩展:支持多标的共享 portfolio 状态
- 目标权重 → 差额下单自动计算
- 滑点/冲击成本模型考虑个股流动性（成交量约束）
- 批量撮合性能:5000 只 × 2400 日 应在 10 秒内完成

### 阶段 11.4 — 因子研究 & 基准归因

- `tradelearn/factor/` 扩充:
  - IC / rank IC / ICIR 时序分析
  - 分位数组合回测（top/bottom N 组）
  - 单因子单调性检验
- 基准对齐:
  - 沪深 300 / 中证 500 / 中证 1000 成分股权重数据接入
  - 超额收益、跟踪误差、信息比率
  - 行业/风格偏离度约束（Barra 风格因子）

### 约束

- 各阶段均不破坏现有 `lite` / `engine` 的逐 bar 策略 API
- 不引入新的顶层入口目录,截面 API 放在 `engine/` 内
- DuckDB 为可选依赖,不影响不使用全市场数据的用户
- 三条不可变量（结果对齐、API 清晰度、实盘扩展边界）继续保持
