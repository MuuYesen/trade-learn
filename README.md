<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="600" />
</p>

<p align="center">
  <strong>Python 写策略与投研流程，Rust 扛事件驱动回测内核。</strong>
</p>

<p align="center">
  <a href="./README_en.md">English version</a>
</p>

**trade-learn** 是一个把 **trade** 和 **learn** 放在同一条链路里的量化框架：面向指数增强、量化研究、机器学习策略和事件驱动回测。Python 保留策略表达、因子研究和模型实验的灵活性，Rust 承担撮合、订单推进、portfolio 计算这类高频回测内核；从研究、回测到报告和实验追踪，都围绕同一套可复现 workflow 展开。

当前 wheel 版本为 **`0.2.0.0`**，对应 trade-learn **2.0 架构**；历史 `0.1.1.8` 仅作为 demo / 1.x 迁移参考。

它想解决的不是"怎么跑一段回测"，而是把一条完整的策略研发链路接起来：

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

trade-learn 的实现路径是先用 Engine 对齐专业 Backtrader 语义，再在同一套 runtime 上构建 Lite。这样 Lite 写起来短，但不是"另起一套简化撮合"：它通过 Engine / Backtrader 这条桥验证底层有效性，也让快速写法和专业写法共享同一套 Stats、撮合和报告口径。

你可以像写 Backtrader 一样写专业策略，也可以用 Lite API 快速验证一个想法；可以接入 TDX、TA-Lib、TradingView、pandas-ta-classic 等指标生态，也可以把因子分析、因果特征筛选、Optuna 参数优化、组合权重、回测报告和实验记录放进同一条工作流。

## 核心亮点

- **Lite 是推荐起点**：更短的写法，适合快速验证、教学、1.x 风格迁移和多资产目标权重；不是另一套撮合逻辑，是同一 runtime 的轻量语法。
- **Backtrader 风格 Engine**：成熟事件驱动模型，适合复杂策略、组合策略、Analyzer / Sizer / Signal 和未来 paper / live adapter。Engine 先对齐 Backtrader，Lite 再复用 Engine 底层内核，这是正确性来源。
- **Rust single / multi-data runner**：单标的是一条数据流逐 bar 推进；多标的 panel 是同一个交易日推进全部 active symbols，适合指数增强和组合调仓。用户仍然只写 `next()`，runner 由数据形态自动选择。
- **性能基线透明**：本机基线（Backtrader 为 1.0x）：
  - 55 万 bar 单标的：Lite 约 **27.9x**、Engine 约 **11.0x**
  - 1000 标的 20 年目标权重：Lite 约 **119.1x**、Engine 约 **69.7x**
- **双市场指标生态**：A 股偏 TDX / MyTT 口径（`tl.tdx` / `bt.tdx`），海外偏 TradingView（`tl.tv` / `bt.tv`）+ pandas-ta-classic / TA-Lib（`tl.pta` / `tl.talib`），口径**显式选择**。
- **Pipeline 投研流水线**：`FeatureSet` / `Pipeline` / `CausalSelector` / `ResearchRun` / `Allocator` 把训练 / 测试 / 预处理 / 评分 / 权重 / 回测连成一条链。
- **因子与报告**：alphalens / pyfolio 风格分析，HTML 报告、交互式 plot、CSV / XLSX artifacts。
- **MLflow / JupyterLab / MCP**：实验追踪、交互研究、自动化工具集成开箱即用。

## 适合谁

- 已经会 Backtrader，但希望有更现代的报告、研究流水线和 Rust 回测内核
- 在做因子研究，希望从 alphalens 风格分析自然走到事件驱动回测
- 在做机器学习策略，希望把 train / test、预处理、因果筛选、评分、权重、回测和 MLflow 记录连起来
- 同时覆盖 A 股和海外市场，不想让指标口径、数据形态和报告体系割裂
- 同时维护规则策略和模型策略，不想让两套策略使用完全不同的数据、报告和实验体系

## 安装

```bash
pip install trade-learn
```

获取最新版本：

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

可选 extras：

| extra | 用途 |
|---|---|
| `[lab]` | JupyterLab / Jupyter AI / MCP / Pygwalker 交互研究环境 |
| `[mlflow]` | MLflow tracking server 与实验 artifact 记录 |
| `[all]` | Lab、MLflow、Riskfolio-Lib、Optuna、DuckDB 等完整研究环境 |
| `[docs]` | 本地构建技术手册网站 |
| `[dev]` | 测试、lint、wheel 构建工具 |

## 快速上手

**Lite——最短路径**（适合快速验证、教学、多资产目标权重）：

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
bt.plot("plot.html")
bt.report("report.html")
```

**Engine——Backtrader 风格**（适合复杂 / 组合策略与未来 paper / live 模式）：

```python
import tradelearn.engine as bt
from tradelearn.data import TradingViewProvider


class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 20))

    def __init__(self):
        self.fast = bt.tdx.MA(self.data.close, N=self.p.fast)
        self.slow = bt.tdx.MA(self.data.close, N=self.p.slow)

    def next(self):
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

## 投研流水线示例

README 只放最短可读版本，完整脚本见 [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) 和 [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py)。

**1. Research：从原始行情生成特征、切分训练集 / 测试集**

```python
import tradelearn.research as research
import tradelearn.research.preprocess as pp

feature_set = research.FeatureSet(
    {
        "alpha": lambda p: p.close.pct_change(20)
        / p.close.pct_change().rolling(20).std(),
        "size": lambda p: p.close,
    },
    target={"label": lambda p: p.close.shift(-20) / p.close - 1.0},
)

features = feature_set.fit_transform(bars, include_target=True).dropna()
train, test = research.time_split(features, split="2023-09-01", level="timestamp")
```

**2. Pipeline：预处理、模型打分、生成权重**

```python
from sklearn.ensemble import GradientBoostingRegressor
import tradelearn.research.portfolio as pf

pipe = research.Pipeline(
    [
        pp.Winsorizer(columns=["alpha"], limits=(0.05, 0.95)),
        pp.Neutralizer(columns=["alpha"], exposures=["size"]),
        pp.StandardScaler(columns=["alpha"]),
    ]
)
train = pipe.fit_transform(train)
test = pipe.transform(test)

model = GradientBoostingRegressor(random_state=7)
model.fit(train[["alpha"]], train["label"])
scores = research.ModelScorer(model, features=("alpha",), current=False).predict(test)

weights = pf.Allocator(
    select=pf.TopK(k=2),
    weight=pf.EqualWeight(gross=0.95),
    constrain=pf.Constraints(max_weight=0.5, normalize=True),
).build(scores)
```

**3. Portfolio：把目标权重交给 Lite / Engine 执行**

```python
class LitePortfolio(tl.Strategy):
    def next(self):
        if len(self.data) % 20 == 0:
            self.target_weights(self.research_result.weights[0], close_missing=True)


test_bars = research.split_bars(bars, split="2023-09-01")
stats = tl.Backtest(test_bars, LitePortfolio, cash=100_000).run(
    research_result=research_result
)
```

**4. Live-style：在策略里只用当前可见窗口推理**

投研流水线适合离线训练和复盘；如果要让策略语义更接近 paper / live，可以把模型和 allocator 放进策略参数，在 `next()` 中用 `history_panel()` 只读取当前已经发生的窗口。

```python
class LiveStylePortfolio(tl.Strategy):
    lookback = 20

    def init(self):
        self.start_on_bar(self.lookback)

    def next(self):
        if len(self.data) % 20 != 0:
            return

        panel = self.history_panel(self.lookback)
        features = self.feature_set.transform(panel).dropna()
        scores = self.scorer.predict(features)
        weights = self.allocator.build(scores)
        self.target_weights(weights, close_missing=True)
```

完整版本：

| 目标 | 完整脚本 |
|---|---|
| Lite 投研 + 回测 + report + MLflow | [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) |
| Engine 投研 + 回测 + report + MLflow | [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) |
| Lite live-style 当前窗口推理 | [`examples/research/index_enhance_lite_live.py`](./examples/research/index_enhance_lite_live.py) |
| Engine live-style 当前窗口推理 | [`examples/research/index_enhance_engine_live.py`](./examples/research/index_enhance_engine_live.py) |
| Engine Backtrader 风格组合调仓 | [`examples/engine/11_target_percent_portfolio.py`](./examples/engine/11_target_percent_portfolio.py) |
| 资产类别组合策略 | [`examples/engine/12_asset_class_portfolios.py`](./examples/engine/12_asset_class_portfolios.py) |


## 对齐与性能

本机基线只看两个问题：结果是否对齐、吞吐是否明显快于 Backtrader。完整复现命令见 [性能基准](./docs/benchmarks.md)。

| 场景 | 引擎 | 数据量 | 耗时 | bars/s | 加速比 | 对齐值 |
|---|---|---:|---:|---:|---:|---:|
| 单标的 SMA | Lite | 550,000 bars | 1.3253s | 414,990 | 27.9x | Final Value 118,399.33 |
| 单标的 SMA | Engine | 550,000 bars | 3.3767s | 162,883 | 11.0x | Final Value 118,399.33 |
| 单标的 SMA | Backtrader | 550,000 bars | 37.0270s | 14,854 | 1.0x | Final Value 118,399.33 |
| 1000 标的 top-50 目标权重 | Lite | 5,040,000 bars | 2.407s | 2,094,237 | 119.1x | Final Value 4,199,638.26 |
| 1000 标的 top-50 目标权重 | Engine | 5,040,000 bars | 4.112s | 1,225,594 | 69.7x | Final Value 4,199,638.26 |
| 1000 标的 top-50 目标权重 | Backtrader | 5,040,000 bars | 286.538s | 17,589 | 1.0x | Final Value 4,199,638.26 |

## 一致性承诺

trade-learn 把"对照基线"当作工程纪律：

- `metrics`（sharpe / max_dd / sortino / ...）对 empyrical：`rtol=1e-10`
- `tl.pta` / `bt.pta` 对 pandas-ta-classic：`rtol=1e-10`
- `tl.tdx` / `bt.tdx` 对 MyTT：`rtol=1e-10`
- `tl.tv` / `bt.tv` 对 pyneCore / TradingView：`rtol=1e-6`
- 回测 **trades**（决策层）对 Backtrader oracle：**0 差异**（时间 / 方向 / size）
- 回测 equity 对 oracle：`rtol=1e-6`，summary：`rtol=1e-4`，每条差异都登记在案

详见 [设计笔记 → 与外部库的语义一致性](docs/internals/consistency.md)。

## 完整文档

完整技术手册（mkdocs 站点）：[`docs/`](./docs/README.md)

| 主题 | 入口                                                                      |
|---|-------------------------------------------------------------------------|
| 30 行走通第一个回测 | [快速开始](./docs/quickstart.md)                                            |
| Lite / Engine 用法 | [Lite 指南](./docs/guides/lite.md) · [Engine 指南](./docs/guides/engine.md) |
| 架构与边界 | [架构](./docs/concepts/architecture.md)                                   |
| 因子 / ML / 权重研究流水线 | [Research 指南](./docs/guides/research.md)                                |
| 双口径指标（`tl.talib` / `tl.pta` / `tl.tdx` / `tl.tv`） | [Indicators 指南](./docs/guides/indicators.md)                            |
| 性能基线 | [性能基准](./docs/benchmarks.md)                                            |
| 内核（契约 / 撮合 / portfolio / 事件循环） | [设计笔记](./docs/internals/contracts.md)                                   |
| 完整 API | [API 参考](./docs/api/reference.md)                                       |

本地预览：

```bash
uv run mkdocs serve
```

## 协议

Apache-2.0。融合的上游署名见 [`NOTICE`](./NOTICE)：empyrical / alphalens / pyfolio / quantstats / MyTT / pandas-ta-classic / pyneCore / causallearn / DolphinDB。backtesting.py 与 backtrader 标注为"inspired by"——不复制源码。

## 致谢

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## 联系方式

微信公众号：知守溪的收纳屋 · 邮箱：muyes88@gmail.com
