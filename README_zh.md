# trade-learn

面向传统量化策略与机器学习策略的 Python 研究框架。

trade-learn 的目标不是再造一个只追求极限跑分的批处理回测器，而是提供一套**事件驱动**的投研到回测闭环：数据获取、指标计算、因子分析、研究流水线、策略回测、报告输出、实验追踪可以在同一个项目语义下协同工作。

底层回测 runtime 使用 Rust 承担撮合、bar loop、订单推进和 portfolio 记账等高频路径；Python 层保留策略表达、指标生态、研究流程和兼容门面。这样既能保持事件驱动策略 API 的清晰度，也能把真正耗时的回测内核下沉。

## 当前定位

trade-learn 2.x 当前定位为：

- **事件驱动回测框架**：以 bar 推进、订单、成交、持仓、资金曲线为主线。
- **Backtrader 风格高级 API**：`tradelearn.engine` 对齐 Backtrader 的 Cerebro / Strategy / Analyzer / Sizer / Signal 心智。
- **Lite 快速研究 API**：`tradelearn.lite` 提供更短的策略写法，适合快速验证、多资产目标权重和 1.x 风格迁移。
- **Python 指标生态优先**：指标不下沉 Rust，统一接入 `talib`、`tdx`、`tv`、`pta` 等 Python 侧指标命名空间。
- **研究流水线**：`tradelearn.research` 提供数据探索、切分、预处理、特征生成、权重构建、实验记录等组件。
- **报告与可视化**：`tradelearn.report` 提供 pyfolio / alphalens 风格分析图和 HTML 报告。
- **实验追踪**：支持 MLflow 记录参数、指标、报告、图表、CSV/XLSX artifacts。

## 设计原则

trade-learn 的实操原则是：

> 以事件驱动为核心，Rust 只承担撮合、bar loop、订单推进、portfolio 等高性能回测内核；指标计算不下沉 Rust，而是通过 TA-Lib、pandas-ta-classic、TDX、TradingView 等 Python 生态工具做批量缓存或 rolling 计算；Python 保持清晰的策略、指标和兼容层边界，core 只放跨 backtest/paper/live 的中性契约，backtest 只放公共回测 runtime，Engine/Lite 专属语义各自留在 facade；所有优化都必须在不破坏结果对齐、策略 API 和未来实盘扩展边界的前提下推进；性能优化服务于事件驱动架构，不为了极限跑分牺牲策略 API 清晰度。

换句话说：

- Rust 是回测内核，不是策略语言。
- Engine/Lite 是用户入口，不是 runtime 实现层。
- `tradelearn.backtest` 是共享回测 runtime，不建议用户直接依赖。
- `tradelearn.core` 只放中性契约，不放 Backtrader、Lite 或回测专属实现。
- 指标、因子、报告、研究工具保持 Python 生态可组合。

## 用户入口

### Engine：Backtrader 风格高级 API

适合复杂策略、组合策略、Analyzer / Observer / Sizer、信号策略、参数优化、未来 paper/live adapter。

```python
import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider


class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 20))

    def __init__(self):
        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)
        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)

    def next(self):
        if self.fast[0] != self.fast[0] or self.slow[0] != self.slow[0]:
            return
        if not self.position and self.fast[0] > self.slow[0]:
            self.buy(size=100)
        elif self.position and self.fast[0] < self.slow[0]:
            self.close()


provider = TradingViewProvider(n_bars=500)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

cerebro = bt.Cerebro(trade_on_close=True)
cerebro.setcash(100_000)
cerebro.setcommission(0.0003)
cerebro.adddata(bars, name="AAPL")
cerebro.addstrategy(SmaCross)

[strategy] = cerebro.run()
print(strategy.stats.summary)
```

### Lite：轻量快速研究 API

适合快速验证、教学、小型策略、多资产目标权重、1.x 风格迁移。

```python
import tradelearn.lite as tl
from tradelearn.data import TradingViewProvider


class LiteSmaCross(tl.Strategy):
    fast = 10
    slow = 20

    def init(self):
        self.fast_ma = tl.tdx.MA(self.data.close, N=self.fast)
        self.slow_ma = tl.tdx.MA(self.data.close, N=self.slow)
        self.start_on_bar(self.slow + 1)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and not self.position():
            self.buy(size=100)
        elif self.fast_ma[0] < self.slow_ma[0] and self.position():
            self.position().close()


provider = TradingViewProvider(n_bars=500)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()
print(stats.summary)
```

## 核心能力

### 1. 事件驱动回测

- Rust 内核负责 exact / smart K 线撮合、订单推进、成交、现金、持仓和权益计算。
- Python 策略保持 `next()`、`buy()`、`sell()`、`close()`、`order_target_percent()` 等事件驱动写法。
- Engine 和 Lite 共用同一套 `tradelearn.backtest` runtime 与 `Stats` 口径。

### 2. 多资产与组合

数据 provider 可以返回单标的 OHLCV，也可以返回 `MultiIndex(timestamp, symbol)` 的 panel。Engine 的 `Cerebro.adddata(panel)` 会自动按 symbol 拆成多个 feed；Lite 的 `Backtest(panel, Strategy)` 也会自动识别多标的。

多资产历史窗口可以使用 tradelearn 增强接口：

```python
panel = self.history_panel(lookback=20)
```

它返回最近已经可见的 OHLCV panel，索引为 `timestamp / symbol`。该接口是 tradelearn 增强能力，不是 Backtrader 原生接口。

### 3. 指标系统

内置指标统一走命名空间：

- `tl.talib` / `bt.talib`：TA-Lib 风格指标。
- `tl.tdx` / `bt.tdx`：TDX / MyTT 口径指标。
- `tl.tv` / `bt.tv`：TradingView / PineCore 口径指标。
- `tl.pta` / `bt.pta`：pandas-ta-classic 口径指标。

同一个指标函数可以接收 tradelearn line，也可以接收 pandas / numpy 数据。传入 line 时会自动包装成对应 facade 的指标对象；传入普通 Series 时返回普通计算结果。

### 4. 因子分析

`tradelearn.factor` 提供 alphalens 风格因子清洗和分析：

```python
from tradelearn.factor import FactorAnalyzer, clean_factor_and_forward_returns

clean = clean_factor_and_forward_returns(
    factors,
    factor="momentum",
    prices=prices,
    periods=(1, 5, 10),
    quantiles=5,
)

fa = FactorAnalyzer.from_clean_factor_data(clean)
fa.report("factor_report.html")
```

单因子和多因子都通过同一套 clean factor data 进入分析器。默认报告可以同时覆盖多个 forward return 周期。

### 5. 研究流水线

`tradelearn.research` 面向投研流程：

- `explore.profile()`：数据概览。
- `split.time_split()` / `split_bars()`：训练期 / 测试期切分。
- `preprocess.Winsorizer / Neutralizer / StandardScaler`：预处理组件。
- `derive.SymbolicFeatureGenerator`：符号特征生成。
- `portfolio.select_top / equal_weight / apply_constraints / WeightBuilder`：组合权重构建。
- `ResearchRun` / `ResearchResult`：记录研究步骤、参数、权重和 artifacts。

研究阶段可以使用完整数据拟合特征、预处理器和模型；回测阶段建议使用 `split_bars()` 只传测试期 bar，避免训练期空仓区间进入策略评估。

### 6. 报告与图表

回测结果通过统一 `Stats` 对象提供：

- `summary`
- `equity`
- `returns`
- `fills`
- `trades`
- `positions`
- `orders`
- `config`

Engine：

```python
[strategy] = cerebro.run()
cerebro.plot("plot.html")
cerebro.report("report.html")
```

Lite：

```python
stats = bt.run()
bt.plot("plot.html")
bt.report("report.html")
```

`Reporter` 可以单独接收收益、持仓和成交，生成 pyfolio 风格报告；`FactorAnalyzer` 生成 alphalens 风格报告。

### 7. MLflow

Lite 和 Engine 都可以记录 MLflow：

- Lite 使用 `Backtest.log_mlflow(...)`，适合 run 后轻量上传。
- Engine 使用 `MLflowAnalyzer`，接入 analyzer 生命周期。

可记录内容包括参数、核心指标、报告 HTML、图表 HTML、CSV artifacts、XLSX artifacts 等。

## 安装

```bash
pip install trade-learn
```

开发版：

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@v2
```

可选能力：

```bash
pip install "trade-learn[mlflow]"
pip install "trade-learn[lab]"
pip install "trade-learn[research]"
pip install "trade-learn[all]"
```

## 项目结构

```text
tradelearn/
  core/        # 跨 backtest / paper / live 的中性契约
  backtest/    # 共享事件驱动回测 runtime
  engine/      # Backtrader 风格高级 API
  lite/        # Tradelearn Lite 快速研究 API
  data/        # 数据 provider 与数据处理
  indicators/  # talib / tdx / tv / pta 指标命名空间
  factor/      # alphalens 风格因子分析
  research/    # 投研流水线、预处理、组合权重构建
  report/      # pyfolio 风格报告与图表
  ml/          # ML 策略与模型工具
  lab/         # JupyterLab / MCP 等研究环境集成
```

## 示例

- `examples/engine/`：Engine / Backtrader 风格策略。
- `examples/lite/`：Lite 轻量策略。
- `examples/research/`：指数增强、研究流水线、MLflow 示例。
- `examples/full_workflow_engine.py`：Engine 完整投研流程。
- `examples/full_workflow_lite.py`：Lite 完整投研流程。

## 文档

- API 与组件调用：`design/COMPONENT_USAGE.md`
- 项目结构：`design/PROJECT_STRUCTURE.md`
- 开发运行手册：`design/RUNBOOK.md`
- 在线文档站点：`mkdocs serve`

## 当前状态

trade-learn 2.x 仍处于快速迭代阶段。稳定主路径是：

1. 使用 `tradelearn.engine` 对齐 Backtrader 风格策略。
2. 使用 `tradelearn.lite` 快速验证和迁移 1.x 风格策略。
3. 使用 `tradelearn.research + factor + report + MLflow` 完成投研闭环。

不建议用户直接依赖 `tradelearn.backtest` 或 `tradelearn.core` 的内部实现细节。
