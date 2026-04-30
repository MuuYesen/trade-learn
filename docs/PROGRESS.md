# Current Progress

最后更新:2026-04-30

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
| 总路线图完成度 | 约 88% |
| 已完成主线 | 工程地基、metrics、factor/report、Rust 撮合核、CLI、JupyterLab + MCP、ML 能力、`engine` 高级 API、`lite` 1.x 低级 API、共享 backtest runtime 与 Rust kernel |
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
| 5 | JupyterLab 环境 | ✅ 完成 | lab dry-run、starter notebooks、doctor、JupyterLab + MCP 联动启动计划已完成 |
| 6 | ML 能力 | ✅ 完成 | engine.MLStrategy、lite.MLStrategy 已闭合；feature_vector() 统一钩子已实现 |
| 7 | MCP Server | ✅ 完成 | `tradelearn mcp` 已提供 stdio / SSE / streamable-http 入口;MCP tools 覆盖 project info、runtime config、lab plan |
| 8 | engine 高级 API | ✅ 完成 | Backtrader 风格 Cerebro/Strategy/feeds/indicators/notify 与迁移策略已闭合;继续补常用接口覆盖 |
| 9 | 文档 + 发版 | ⏸️ 暂缓 | 文档、release golden gate、examples 策略目录整理已完成;wheel/PyPI/GitHub Release 等发版闭环暂不优先 |
| 10 | QMT 实盘接口 + Rust BarRunner | 🟡 收口 | QMT 具体文件暂不提交;事件驱动实盘兼容接口保留;性能优化只做 profile-driven 小修 |
| 11 | 指数增强 / 全市场截面 | ✅ 主线完成 | DuckDB backend、engine 截面 rebalance、Lite target weights、多资产 portfolio benchmark、FactorAnalyzer IC/分位/报告、pyfolio-style 多因子风险/收益归因已闭合;qlib 暂缓 |

## 当前验证入口

```bash
uv run pytest tests/unit/lite tests/unit/examples/test_1x_strategy_examples.py -q
uv run pytest tests/unit/backtest/test_benchmark_runner_scripts.py -q
uv run pytest tests/unit/backtest/test_core_layering.py -q
uv run python benchmarks/runners/benchmark_bt.py
uv run python benchmarks/runners/benchmark_throughput.py --bars 550000 --repeat 1 --warmup 0
uv run pytest tests/unit/factor -q
```

## 最近关键提交

- `87f6a20` 共享 feed 构造下沉到 backtest runtime,engine/lite 不互相依赖。
- `581391e` 拆分 Lite 策略支撑模块,`strategy.py` 保持薄 facade。
- `5123b6c` 清理 Lite facade 边界,移除 `Backtesting*` 命名残留并改用 broker 公共 storage API。
- `b32d24e` 记录 55 万 bar throughput 与 profiling 口径。
- `8f757be` 统一 throughput benchmark 语义,Engine/Lite/Backtrader 输出 Final Value / Fills / Closed Trades 一致,并修复 Lite `trade_on_close` 透传。
- `6234b51` / `f7fef79` 补齐 Backtrader broker / Cerebro 常用薄别名:`set_cash/get_cash/get_value/getbroker/setbroker/addcommissioninfo`。
- `d79d7cd` 补齐 Backtrader 查询型 surface:`addstore`、`datasbyname`、`get_orders_history/get_orders_open`、`getpositionbyname`。
- `9d12a59` 补齐 JupyterLab + MCP 集成:`tradelearn lab` 启动 JupyterLab 与 MCP HTTP server;`tradelearn mcp` 支持 stdio / SSE / streamable-http。
- `ae9902f` 完成 Stage 11 截面工作流主线:`IndexEnhanceStrategy`、DuckDB/target_weights benchmark、factor/report 接入。
- `579e48f` 增加 pyfolio-style factor attribution:`FactorRiskModel` 与 `PerformanceAttribution`。
- `12bc9bb` 补齐 clean-room Alphalens-style factor 主线:`clean_factor_and_forward_returns`、group-neutral、monthly IC、event returns。

## 当前待办

1. 功能优先:继续补齐 `engine` 作为 Backtrader 风格高级 API 的常用接口覆盖;只接受能完整落地的薄接口,不做半语义 stub。
2. Lite 继续保持 1.x 简洁语法,不扩成 `backtesting.py`;新增订单/broker 语义若影响共享 runtime,需要同步提供 Lite 轻量翻译。
3. 正确性门禁:`benchmark_bt.py` 保持 8/8 EXACT;Lite smoke 与 1.x examples 持续通过;`benchmark_throughput.py` 保持 Final Value / Fills / Closed Trades 口径一致。
4. Stage 10 性能优化进入收口状态;后续只接受 profile 证明的局部小修,暂不推进独立 fast runner、core slim mode、批 callback、DSL 或策略下沉。
5. 清理类任务: 后续可拆分超大测试文件 `test_rust_exact_matching.py`、`test_strategy_api.py`、`test_alpha_metadata.py`。
6. Stage 9 发版闭环:wheel 含 Rust 二进制、PyPI / GitHub Release / NOTICE 最终审查暂缓,待功能补齐后恢复。
7. ~~**[P1]** `MLStrategy` 重构：`feature_vector()` 统一钩子替代现行字符串列名方式，消除训练/推理特征不一致隐患。~~ ✅ 已完成
8. ~~**[P1]** `FunctionIndicator.on_bar` 实现，实盘/paper 模式逐 bar 指标更新依赖此接口。~~ ✅ 已完成（buffer 重算方案，O(n)，适合 paper/live 场景）
9. ~~**[P1]** Lite 组合调仓 API 接入共享订单路径。~~ ✅ 已完成（`target_percent/target_weights/target_equal/close_all` → `order_target_value`; `Allocation.rebalance()` 已删除）
10. ~~**[P2]** Signal known-bug 清单专项复核。~~ ✅ 已完成（`SIGNAL_LONG_ANY/SHORT_ANY` 持续信号不再反复平仓; 非数值信号按 neutral 处理; 回归覆盖）

## 功能缺口（按优先级）

### P1 — 用户直接可见

- Lite 目标组合 API 已接入：`target_percent()`、`target_weights()`、`target_equal()`、`close_all()`；`Allocation.rebalance()` 不再作为用户入口。
- `signal.py` known-bug 专项复核已完成；当前已有 SignalStrategy 基础、ANY 持续信号、非数值 neutral 回归测试。

### P2 — engine Backtrader 接口

- `notify_order` / `notify_trade` / `notify_cashvalue` 已有回归覆盖，仍需在完整 benchmark 回归中持续守住。
- `optstrategy` 优化结果访问方式已具备基础 grid run，Backtrader 式结果体验仍可继续打磨。
- `cerebro.plot()` 已定位为行情回放图；是否继续把 analyzer 数据（DrawDown、SharpeRatio 等）叠进图表，按实际用户需求推进。

### P3 — factor / report 层

- `FactorAnalyzer` `.plot()` / `.html()` 已实现。
- 因子单调性检验 `monotonicity()` 已实现。
- Alpha101 / Alpha191 当前元数据与测试口径均已补齐:101/101 supported、191/191 supported、0 skipped。
- clean-room Alphalens-style 主线已完成:`clean_factor_and_forward_returns`、group IC、group-neutral quantile returns、monthly IC heatmap、event-window returns。

### P4 — 暂缓

- Barra-like 风险模型已改为 pyfolio-style clean-room 归因框架：支持用户传入 exposures / factor covariance / specific risk / factor returns，不声称实现商业 Barra 口径。
- qlib 兼容暂缓：当前 project 定位优先保持 engine / lite / backtest runtime 清晰。

## plot / report 架构决策（2026-04-30）

```
cerebro.plot() / bt.plot()     # 行情回放图（backtrader 兼容入口）
stats.plot()                   # 同上，从 result 对象调用

stats.report("report.html")   # 一键完整报告（薄 wrapper → Reporter）
Reporter(stats).html(...)      # 高级定制入口

# Reporter 高级用法
Reporter(stats, benchmark="000300").html(...)          # 加基准
Reporter(stats, market_data=data).html(...)            # 含行情图
Reporter([stats_a, stats_b], names=[...]).html(...)    # 多策略对比
Reporter(stats).tearsheet()                            # notebook inline
Reporter(stats).excel("report.xlsx")                  # Excel 输出
```

规则：
- `plot()` 属于 backtest facade，画策略执行过程（行情 + 成交点 + 指标）
- `Reporter` 是 report 层唯一正式 HTML 报告入口
- `Cerebro.html()` / `Backtest.html()` 不作为推荐 API，最多是 deprecated wrapper
- backtrader 迁移用户用 `cerebro.plot()`，无需改心智模型

## 第三方兼容决策（2026-04-30）

- **NautilusTrader Cython 指标体系**：不采用，回测场景向量化预计算已更快，且 Cython 会破坏 Python 生态兼容性
- **指标加速路径**：优先 Numba JIT 热点函数（P1），其次 Rust 可选后端通过 `FunctionIndicator._func` 注入（P2）

## MLStrategy 架构决策（2026-04-30）

### 分层设计

一套实现，两套入口：

```
tradelearn.backtest.ml_strategy.CoreMLStrategy   # 共享逻辑
    ├── tradelearn.ml.MLStrategy                 # engine 用
    └── tradelearn.lite.MLStrategy               # lite 用
```

engine/lite 差异只有指标写法，ML 核心逻辑（训练、预测、下单）全部下沉 CoreMLStrategy：

| | Engine MLStrategy | Lite MLStrategy |
|---|---|---|
| 继承自 | `engine.Strategy` | `lite.Strategy` |
| 指标写法 | `__init__` + `tl.tdx.X(self.data.close)` | `init()` + `self.I(tl.tdx.X, ...)` |
| 其他所有逻辑 | ← 共享 CoreMLStrategy → | |

### 用户写法

```python
# Engine
from tradelearn.ml import MLStrategy
import tradelearn as tl

class MyML(MLStrategy):
    model = GradientBoostingClassifier()

    def __init__(self):
        self.rsi     = tl.tdx.RSI(self.data.close, n=14)
        self.ma_fast = tl.tdx.MA(self.data.close, n=5)
        self.ma_slow = tl.tdx.MA(self.data.close, n=20)

    def feature_vector(self):
        return {
            "rsi":     self.rsi[0],
            "ma_diff": self.ma_fast[0] - self.ma_slow[0],
        }

    def target(self, data):
        return (data.close.series.shift(-1) > data.close.series).astype(int)

# Lite
from tradelearn.lite import MLStrategy

class MyML(MLStrategy):
    model = GradientBoostingClassifier()

    def init(self):
        self.rsi     = self.I(tl.tdx.RSI, self.data.close, n=14)
        self.ma_fast = self.I(tl.tdx.MA, self.data.close, n=5)
        self.ma_slow = self.I(tl.tdx.MA, self.data.close, n=20)

    def feature_vector(self):
        return {
            "rsi":     self.rsi[0],
            "ma_diff": self.ma_fast[0] - self.ma_slow[0],
        }

    def target(self, data):
        return (data.close.shift(-1) > data.close).astype(int)
```

### 规则

- `__init__` / `init` 指标写法与普通策略完全一致，心智模型零增量
- `feature_vector()` 逐 bar 取 `[0]`，框架自动构建训练矩阵
- `target()` 批量返回标签 Series，框架在 `start()` 自动 fit
- alpha101/191 在 `__init__` / `init` 批量预算好，`feature_vector()` 直接取 `[0]`

### 场景一：tradelearn 端到端训练（推荐）

不依赖 qlib，特征和训练都在 tradelearn 里完成：

```python
from tradelearn.lite import Backtest, MLStrategy
from sklearn.ensemble import GradientBoostingClassifier
import tradelearn as tl

class MyML(MLStrategy):
    model = GradientBoostingClassifier()  # 未训练，start() 自动 fit

    def init(self):
        self.rsi     = self.I(tl.tdx.RSI, self.data.close, n=14)
        self.ma_fast = self.I(tl.tdx.MA, self.data.close, n=5)
        self.ma_slow = self.I(tl.tdx.MA, self.data.close, n=20)

    def feature_vector(self):
        return [self.rsi[0], self.ma_fast[0] - self.ma_slow[0]]

    def target(self, data):
        return (data.close.shift(-1) > data.close).astype(int)

stats = Backtest(df, MyML, cash=100_000).run()
```

### 自定义下单逻辑

`apply_prediction()` 可覆写，完全控制订单类型和细节：

```python
class MyML(MLStrategy):
    model = ...

    def feature_vector(self):
        ...

    def apply_prediction(self, prediction: float):
        if prediction > 0.6:
            self.buy(
                size=self.equity * 0.1 / self.data.close[0],
                exectype=Order.Limit,
                price=self.data.close[0] * 0.99,
            )
        elif prediction < 0.4 and self.position:
            self.close()
```

默认 `apply_prediction()` 只支持市价单 + 固定 size；需要限价单、止损单、bracket order、仓位百分比等，覆写此方法即可。

## Stage 11: 指数增强 / 全市场截面策略支持（主线完成）

面向全市场股票（A 股 5000+）指数增强场景,分四阶段推进:

### 阶段 11.1 — DuckDB 数据 backend ✅

- 在 `tradelearn/data/` 下新增 DuckDB 存储 backend,作为 `pd.read_csv` 的可选替代
- 支持列式裁剪（只拉策略需要的列）和日期范围懒加载
- 全市场日频 10 年（~1200 万行）能在秒级完成加载
- DuckDB 作为可选依赖（`pip install tradelearn[duckdb]`）

### 阶段 11.2 — 截面再平衡 API ✅

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

### 阶段 11.3 — 多标的 portfolio 撮合 ✅

- 现有 backtest runtime + Rust broker 已支持多 data 共享 portfolio 状态。
- `lite.target_weights()` 与 `engine.IndexEnhanceStrategy.target_weights()` 统一走目标权重 → 差额订单 → broker 执行链路。
- 已有 `benchmark_target_weights.py` 输出 target_weights / order_submit / stats_read 分段耗时。
- 冲击成本和成交量约束不默认开启，后续必须有可验证模型与 oracle 后再推进。

### 阶段 11.4 — 因子研究 & 基准归因 ✅ / 部分暂缓

- `tradelearn/factor/` 扩充:
  - IC / rank IC / ICIR 时序分析
  - 分位数组合回测（top/bottom N 组）
  - 单因子单调性检验
  - clean-room Alphalens-style 主线:
    - `clean_factor_and_forward_returns`: factor / prices / forward returns / quantile / group 清洗
    - group IC、group-neutral quantile returns、monthly IC heatmap
    - event-window average returns
  - pyfolio-style 多因子风险模型与收益归因:
    - `FactorRiskModel`: 组合因子暴露、总风险、主动风险、风险贡献
    - `PerformanceAttribution`: factor/common/specific/total returns 拆解
    - 支持用户传入 Barra-like exposures / factor covariance / specific risk,但不内置或声明商业 Barra 口径
- 基准对齐:
  - 沪深 300 / 中证 500 / 中证 1000 成分股权重数据接入
  - 超额收益、跟踪误差、信息比率
  - 行业/风格偏离度约束后续基于 `FactorRiskModel` 接入,具体 optimizer 约束需等待可验证 oracle

### 约束

- 各阶段均不破坏现有 `lite` / `engine` 的逐 bar 策略 API
- 不引入新的顶层入口目录,截面 API 放在 `engine/` 内
- DuckDB 为可选依赖,不影响不使用全市场数据的用户
- 三条不可变量（结果对齐、API 清晰度、实盘扩展边界）继续保持
