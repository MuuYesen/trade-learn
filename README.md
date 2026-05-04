<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>官方文档</b></a> |
  <a href="./README_en.md"><b>English</b></a> |
  <a href="./README_ja.md"><b>日本語</b></a>
</p>

<p align="center">
  <strong>Python 写策略与投研流程，Rust 扛事件驱动回测内核。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v1.0.0-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="Code style">
</p>

**trade-learn** 旨在解决量化投研中研究（Learn）与回测（Trade）脱节的痛点。它通过「Python 表达逻辑 + Rust 原生内核」的混合架构，在确保逻辑与 Backtrader **100% 严苛对齐** 的基础上，实现了多资产回测 **110x+** 的性能飞跃，将大规模验证的耗时从小时级缩短至秒级，为指数增强与机器学习策略提供极速的迭代体验。

极致性能之外，项目也同样关注研究的科学性。针对机器学习策略中常见的“伪相关”风险，我们将 **因果推断 (Causal Inference)** 深度集成进投研流程，通过识别真实的因果驱动路径，有效降低样本外衰减，助您构建具备强解释性与稳健性的量化系统。

为了让这些科学的方法论真正落地，**trade-learn** 并不只是一个高性能引擎，它更是一套完整且按需内置 **JupyterLab** 与 **MLflow** 的全生命周期投研流水线。它将因子挖掘、策略验证与实验审计有机整合为一套完整的全生命周期投研流水线，确保每一个研究决策都可回溯、可审计，让研究员能够真正专注于核心策略的开发与快速迭代。

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## 实现路径

**trade-learn** 拒绝功能的简单堆砌，而是通过独特的“双模双核”设计，在专业深度与研发效率之间构建了平衡。底层通过 **Engine** 深度对齐 Backtrader 语义以夯实逻辑正确性，上层则由 **Lite** 提供极简的 Pythonic 接口，确保了“快”与“准”的统一。

您可以根据研发阶段，自由定义策略的深度：
- **Engine 模式 (深度研发)**：深度对齐 Backtrader 语义，支持 Analyzer/Sizer/Signal 完整生态，适合构建逻辑精密、颗粒度极细的生产级复杂系统。
- **Lite 模式 (敏捷验证)**：沿袭 backtesting.py 的极简主义，支持模型权重直连，极其适合在因子挖掘阶段进行高频迭代与原型验证。

它不仅无缝兼容 TA-Lib、Pandas-TA-Classic、TDX、TradingView 等主流指标库，更创造性地将**因果推断 (Causal Inference)** 引入因子研究。通过内置的 `CausalSelector`，项目将特征筛选、参数寻优与回测报告有机连接，为您呈现一条闭环、透明且高效的量化投研流水线。

## 核心亮点

#### ⚡️ 高能内核：Rust 驱动的极致性能
- **Rust 混合动力**：撮合引擎与核心计算由 Rust 承载，提供单标的 **28x**、多资产调仓 **110x+** 的 Backtrader 级加速。
- **自动 Runner 调度**：根据数据形态自动选择“单流逐 Bar”或“Panel 批量”推进。**针对指数增强场景优化了内存布局**，开发者只需关注 `next()` 逻辑。

#### 🛡️ 严谨金融：Backtrader 语义 100% 对齐
- **Engine 级对齐**：完整支持 Analyzer / Sizer / Signal 体系，确保回测 Trades 与 Backtrader Oracle 逻辑零差异，高度支持自拓展组件。
- **Lite 极简表达**：在同一 Runtime 上构建的轻量语法。**内置 `target_weights` 接口**，将机器学习模型输出的权重一键转化为回测决策。

#### 🧪 因果投研：跨越相关性的科学流程
- **Causal-First 特征筛选**：内置 PC / FCI 等因果发现算法，识别因子的真实驱动路径，从源头对抗回测中的“伪相关”与过拟合。
- **Pipeline 全链路流水线**：将特征工程、因果筛选、评分模型、组合权重与回测报告无缝耦合，形成可复现的实验闭环。

#### 📦 模块化平台：轻量核心，按需扩展
- **核心解耦**：默认安装仅包含高性能回测内核，极简依赖，方便集成至服务器或自动化交易系统。
- **弹性扩展**：通过 `[lab]` 或 `[all]` 扩展，可一键激活 **JupyterLab + MLflow + AI 助手** 组成的集成投研环境，实现“按需加载、随处运行”。


#### 🌍 全球视野：多口径指标与现代生态
- **双市场口径**：显式支持 TDX (A股) / TradingView (海外) 指标口径，深度兼容 TA-Lib 与 pandas-ta。
- **现代投研工具**：开箱即用的 HTML 交互式报告、MLflow 实验追踪以及 JupyterLab / MCP 深度集成。

## 因果投研：跨越“伪相关”陷阱

大多数量化研究止步于**统计相关性 (Correlation)**，这极易导致因子在回测中表现优异、实盘中却迅速失效（过拟合）。trade-learn 通过内置的 **因果发现 (Causal Discovery)** 机制，帮助您识别收益背后的真实动因：

- **因果特征筛选**：通过 `CausalSelector` 结合 PC / FCI 算法，剥离由于“共同观测”产生的伪相关因子，仅保留对收益具有直接驱动能力的特征。
- **抵抗样本外衰减**：基于因果图定位的 Alpha 因子在市场风格切换时具备更强的生存力，有效降低从研究到实盘的性能落差。
- **工业级无缝集成**：深度整合 `causal-learn` 生态，让前沿的因果推断技术像调用 `corr()` 一样丝滑，极大降低了学术算法的落地门槛。

## 适合谁

*   **⚡️ 敏捷开发者与灵感验证**
    厌倦了厚重的配置，希望在几行代码内完成从想法到回测报告的转化，享受类 backtesting.py 的轻快体验，极其适合快速验证原型。
*   **📈 指数增强与组合管理**
    面对 1000+ 标的的大规模回测，利用 Rust Panel Runner 实现秒级调仓模拟，彻底告别传统框架在多资产处理上的漫长等待。
*   **🧠 机器学习与因子研究**
    希望将特征工程、**因果发现**、模型训练（MLflow 追踪）与回测一站式打通，构建从数据到报告的完整自动化闭环。
*   **🛠️ Backtrader 资深玩家**
    在保留成熟事件驱动语义的同时，寻求更现代的报告体系、全链路流水线以及高性能 Rust 回测内核。
*   **🌐 跨市场与多策略团队**
    *   **跨市场统一**：同时覆盖 A股 (TDX) 与海外 (TradingView)，要求指标口径与报告体系完全一致。
    *   **全体系维护**：统一管理规则策略与模型策略，拒绝工具链割裂带来的研发与维护成本。
*   **🔍 因果推断探索者**
    致力于在因子筛选阶段引入因果图技术，通过剔除“伪相关”来构建具有强解释性和高稳健性的量化系统。

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

> **💡 安装建议**：
> 默认安装仅包含核心回测引擎。若需开启包含 JupyterLab 与 MLflow 的全栈投研体验，请指定 `[all]` 扩展进行安装：
> ```bash
> pip install "trade-learn[all]"
> ```
> 在项目根目录，使用命令行启动 `tradelearn lab` 后，默认可通过 `8888` 端口进入交互式环境，通过 `5050` 端口查看 MLflow 实验记录。



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
bt.plot()
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

cerebro.plot()
cerebro.report("report.html")
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

投研流水线适合离线训练和复盘；如果要让策略语义更接近实盘，可以把模型和 allocator 放进策略参数，在 `next()` 中用 `history_panel()` 只读取当前已经发生的窗口。

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

本机基线关注两个核心：**结果是否对齐**、**吞吐是否明显快于 Backtrader**。完整复现命令见 [性能基准](./docs/benchmarks.md)。

#### 1. 单标的高频压测：双均线交叉 (55 万 Bar)
* **策略原理**：执行标准的双均线交叉逻辑。旨在压测 Rust 在处理长序列、单数据流时的事件驱动性能与状态维护效率，挑战单核推进极限。

| 引擎 | 耗时 | bars/s | **加速比** | 对齐值 |
|---|---:|---:|---:|---:|
| **Lite** | 1.32s | **414,990** | **27.9x** | Final Value 118,399.33 |
| **Engine** | 3.37s | **162,883** | **11.0x** | Final Value 118,399.33 |
| Backtrader | 37.02s | 14,854 | 1.0x | Final Value 118,399.33 |

#### 2. 多标的大规模指增：Top-50 目标权重 (504 万 Bar)
* **策略原理**：模拟 1000 标的全市场选股调仓。旨在压测 Rust 对大规模 Panel 数据的内存布局优化与并发处理能力，真实还原机器学习策略的投研场景。

| 引擎 | 耗时 | bars/s | **加速比** | 对齐值 |
|---|---:|---:|---:|---:|
| **Lite** | 2.40s | **2,094,237** | **119.1x** | Final Value 4,199,638.26 |
| **Engine** | 4.11s | **1,225,594** | **69.7x** | Final Value 4,199,638.26 |
| Backtrader | 286.53s | 17,589 | 1.0x | Final Value 4,199,638.26 |

## 一致性承诺

**trade-learn** 将“对照基线”视为核心工程纪律。我们确保每一项计算结果都经得起严苛推敲，并在以下维度保持数值对齐：

*   **金融指标对齐**：`metrics`（Sharpe, MaxDD, Sortino 等）完全对标 `empyrical`，误差控制在 `rtol=1e-10`。
*   **多源指标对齐**：
    *   `tl.pta` (经典指标) 对标 `pandas-ta-classic`：`rtol=1e-10`。
    *   `tl.tdx` (通达信口径) 对标 `MyTT`：`rtol=1e-10`。
    *   `tl.tv` (TradingView 口径) 对标 `pyneCore` 实现：`rtol=1e-6`。
*   **回测引擎对齐**：
    *   **决策层**：成交记录 (**Trades**) 对标 Backtrader 官方实现，实现 **0 差异**（时间、方向、头寸完全一致）。
    *   **净值层**：Equity 曲线误差 `rtol=1e-6`，汇总统计数据误差 `rtol=1e-4`。

> [!IMPORTANT]
> 我们对每一处数值微差都持有“零容忍”态度，所有偏差均登记在案并提供原因分析。详见 [设计笔记 → 语义一致性审计](docs/internals/consistency.md)。

## 完整文档

*   **官方在线文档**：[**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **本地技术手册**：[`docs/`](./docs/README.md)

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

## 🚀 路线图 (Roadmap)

*   **v1.0.x (Stable Release - 当前阶段)**
    *   [x] 基于 Rust 的多标的 Clocked Runner (Stage 13)。
    *   [x] 完整的指数增强研发流水线 (Research -> Weight -> Backtest)。
    *   [x] 深度集成 MLflow 实验追踪与 HTML 现代化报告。
*   **v1.1.x (Advanced Research)**
    *   [ ] **因果推断增强**：集成更多因果图算法（如 GIES、Direct-LiNGAM），提供更强的因子可解释性。
    *   [ ] **高性能连接器**：直连 DolphinDB 与 DuckDB 原生存储，实现亿级 Bar 的秒级读取。
    *   [ ] **更多风险模型**：引入 Barra 风格风险暴露分析与超额收益分解。
*   **v1.2.x (Live & Production)**
    *   [ ] **实盘适配器**：开放通用实盘事件接口，支持 QMT (国金/华宝) 等券商柜台接入。
    *   [ ] **分布式参数优化**：基于 Ray/Optuna 的多机并行参数搜索。
    *   [ ] **Agent 深度集成**：通过 MCP 协议实现 LLM 对投研流水线的自动化控制。

## 致谢

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## 联系方式

微信公众号：知守溪的收纳屋 · 邮箱：muyes88@gmail.com
