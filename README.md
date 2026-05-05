<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>官方文档</b></a> |
  <a href="./CHANGELOG.md"><b>更新日志</b></a> |
  <a href="./README_en.md"><b>English</b></a> |
  <a href="./README_ja.md"><b>日本語</b></a>
</p>

<p align="center">
  <strong>Python 写策略与投研流程，Rust 扛事件驱动回测内核。</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.4-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Changelog-v0.2.4-blue?style=flat-square" alt="Changelog">
</p>

**trade-learn** 旨在解决量化投研中「研究（Learn）」与「回测（Trade）」长期脱节的问题。它采用「Python 表达策略逻辑 + Rust 原生回测内核」的混合架构，在与 Backtrader **100% 严格语义对齐** 的前提下，实现多资产回测 **110x+ 的性能提升**，将大规模策略验证从小时级压缩至秒级，为指数增强与机器学习策略提供真正可迭代的研究效率。

在高性能之外，**trade-learn** 同时提供一套完整的投研基础设施。框架内置 **JupyterLab** 与 **MLflow**，将因子挖掘、策略验证与实验审计无缝串联，构建出一条**可复现、可追踪、可审计**的全生命周期投研流水线，使研究过程从“结果导向”升级为可系统化管理的工程流程，让研究员专注于策略逻辑本身。。

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

在方法论层面，项目同样强调研究的科学性。针对机器学习策略中常见的“伪相关”问题，**trade-learn** 将 **因果推断（Causal Inference）** 深度融入投研流程，通过识别真实的因果驱动路径，降低样本外衰减风险，帮助构建兼具解释性与稳健性的量化策略体系。

## 实现路径

**trade-learn** 拒绝功能的简单堆砌，而是通过独特的“双模双核”设计，在专业深度与研发效率之间构建了平衡。底层通过 **Engine** 深度对齐 Backtrader 语义以夯实逻辑正确性，上层则由 **Lite** 提供极简的 Pythonic 接口，确保了“快”与“准”的统一。

您可以根据研发阶段，自由定义策略的深度：
- **Engine 模式 (深度研发)**：深度对齐 Backtrader 语义，支持 Analyzer/Sizer/Signal 完整生态，适合构建逻辑精密、颗粒度极细的生产级复杂系统。
- **Lite 模式 (敏捷验证)**：沿袭 backtesting.py 的极简主义，支持模型权重直连，极其适合在因子挖掘阶段进行高频迭代与原型验证。

在生态层面，**trade-learn** 提供完善的指标支持。框架兼容 TA-Lib、Pandas-TA-Classic、TDX、TradingView 等主流指标库，并支持灵活扩展自定义指标与数据源，满足不同研究场景的需求。

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
- **双市场口径**：显式支持 TDX (国内) 和 TradingView (国外) 指标口径，同时深度兼容 TA-Lib 与 Pandas-TA-Classic。
- **现代投研工具**：开箱即用的 HTML 交互式报告、MLflow 实验追踪以及 JupyterLab / MCP 深度集成。

## 因果投研：跨越“伪相关”陷阱

大多数量化研究止步于**统计相关性 (Correlation)**，这极易导致因子在回测中表现优异、实盘中却迅速失效（过拟合）。trade-learn 通过内置的 **因果发现 (Causal Discovery)** 机制，帮助您识别收益背后的真实动因：

- **因果特征筛选**：通过 `CausalSelector` 结合 PC / FCI 算法，剥离由于“共同观测”产生的伪相关因子，仅保留对收益具有直接驱动能力的特征。
- **抵抗样本外衰减**：基于因果图定位的 Alpha 因子在市场风格切换时具备更强的生存力，有效降低从研究到实盘的性能落差。
- **工业级无缝集成**：深度整合 `causal-learn` 生态，让前沿的因果推断技术像调用 `corr()` 一样丝滑，极大降低了学术算法的落地门槛。

## 适合谁

*   **敏捷开发者与灵感验证**：
    厌倦了厚重的配置，希望在几行代码内完成从想法到回测报告的转化，享受类 backtesting.py 的轻快体验，极其适合快速验证原型。
*   **指数增强与组合管理**：
    面对 1000+ 标的的大规模回测，利用 Rust Panel Runner 实现秒级调仓模拟，彻底告别传统框架在多资产处理上的漫长等待。
*   **机器学习与因子研究**：
    希望将特征工程、**因果发现**、模型训练（MLflow 追踪）与回测一站式打通，构建从数据到报告的完整自动化闭环。
*   **Backtrader 资深玩家**：
    在保留成熟事件驱动语义的同时，寻求更现代的报告体系、全链路流水线以及高性能 Rust 回测内核。
*   **跨市场与多策略团队**：
    *   **跨市场统一**：同时覆盖 A股 (TDX) 与海外 (TradingView)，要求指标口径与报告体系完全一致。
    *   **全体系维护**：统一管理规则策略与模型策略，拒绝工具链割裂带来的研发与维护成本。
*   **因果推断探索者**：
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
> 在项目根目录使用命令行启动 `tradelearn lab` 后，默认可通过 `8888` 端口进入交互式环境，通过 `5050` 端口查看 MLflow 实验记录。



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


provider = TradingViewProvider(n_bars=5000)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()

print(stats.summary)
bt.plot()
bt.report("report.html")
```

> [!TIP]
> **关于多标的逻辑：** 在多标的回测场景下，策略默认会绑定到 `self.data`（主数据源）。这意味着上述代码即使传入了多个标的，也仅会根据第一个标的的信号进行决策。若要实现多标的独立并行交易，需在策略 `init` 中遍历 `self.datas` 为每个标的建立指标。

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


provider = TradingViewProvider(n_bars=5000)
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

> [!TIP]
> **关于多标的逻辑：** 在多标的回测场景下，策略默认会绑定到 `self.data`（主数据源）。这意味着上述代码即使传入了多个标的，也仅会根据第一个标的的信号进行决策。若要实现多标的独立并行交易，需在策略 `init` 中遍历 `self.datas` 为每个标的建立指标。

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

| 引擎模式 | 处理耗时 | 吞吐量 (Bars/s) | **加速比** | 最终权益 | 成交单数 | 闭环交易 | 状态 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 | - |

#### 2. 多标的大规模指增：Top-50 目标权重 (504 万 Bar)
* **策略原理**：模拟 1000 标的全市场选股调仓。旨在压测 Rust 对大规模 Panel 数据的内存布局优化与并发处理能力，真实还原机器学习策略的投研场景。

| 引擎模式 | 处理耗时 | 吞吐量 (Bars/s) | **加速比** | 最终权益 | 完成订单 | 调仓意图 | 重平衡次数 |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **2.40s** | **2,094,237** | **119.1x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| **Tradelearn Engine** | **4.11s** | **1,225,594** | **69.7x** | **4,199,638.26** | 23,249 | 23,249 | 239 |
| Backtrader (Oracle) | 286.53s | 17,589 | 1.0x | 4,199,638.26 | 23,249 | 23,249 | 239 |

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

基于当前的工程规划，我们将 **trade-learn** 的进化路径划分为以下核心维度：

#### 回测引擎与核心地基
- [x] **Rust 混合内核**：实现 Clocked Multi-Data Runner，多资产回测性能提升 **110x+**。
- [x] **Backtrader 语义对齐**：撮合逻辑 100% 一致，支持通过 `bt.Strategy` 共享底层 Runtime。
- [x] **指数增强流水线**：打通 `Data → Factor → Score → Weights → target_weights()` 全流程。
- [x] **自动化实验审计**：深度集成 MLflow，自动记录代码快照、参数、指标及报告。
- [x] **高性能数据后端**：**DuckDB 原生连接器已落地**，支持亿级 Bar 的本地秒级读取与跨维度查询。
- [ ] **风险模型集成**：支持 Barra 风格风险暴露分析与超额收益分解。

#### 科学投研能力
- [x] **因果发现基础**：集成 `CausalSelector` (PC/FCI)，从特征工程阶段识别 Alpha 真实驱动。
- [ ] **算法增强**：引入 GIES、Direct-LiNGAM 等高级算法，提升因子筛选的可解释性与稳定性。
- [ ] **因果驱动闭环**：将因果分析与参数优化、风险控制形成自动化的研发闭环。

#### 智能体与 AI 能力
- [x] **MCP 知识网关**：**MCP Server 已上线**，实现 AI 对 API 的结构化理解与自动代码生成。
- [ ] **Agentic 策略诊断**：利用 LLM 自动解析回测结果，识别亏损动因并给出逻辑优化建议。
- [ ] **LLM 因子解释器**：利用大模型将因果发现结果转化为直观的金融投资逻辑。

#### 工程化与 ML 生命周期
- [x] **Model Registry**：**基于 MLflow 实现模型注册**，支持特征指纹与模型版本的全生命周期管理。
- [ ] **分布式参数优化**：基于 Ray / Optuna 扩展多机并行的参数搜索与蒙特卡洛模拟。

#### 实盘与生态愿景
- [x] **通用实盘事件链路**：完成 `EventRunner` 语义，支持回测与实盘代码 100% 复用。
- [ ] **实盘交易打通**：对接 `QMT`, `IBKR` 等券商接口，完成从研究到执行的最后一公里。
- [ ] **Agentic Quant 平台**：最终演进为支持自然语言驱动的全流程投研自动化底座。

## 免责声明 (Disclaimer)

本项目仅用于学术研究与技术交流，不构成任何投资建议。量化交易存在极高风险，回测表现不代表未来实盘收益。开发者不对因使用本项目导致的任何经济损失承担责任。请投资者谨慎决策，风险自担。

*This project is for academic research and technical exchange only and does not constitute any investment advice. Quantitative trading involves high risk; past performance is not indicative of future results. The developers are not responsible for any financial losses incurred through the use of this project. Invest at your own risk.*

## 致谢

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## 联系方式

微信公众号：知守溪的收纳屋 · 邮箱：muyes88@gmail.com
