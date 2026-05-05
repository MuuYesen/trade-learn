<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>Documentation</b></a> |
  <a href="./README.md"><b>中文版</b></a> |
  <a href="./README_ja.md"><b>日本語</b></a>
</p>

<p align="center">
  <strong>Python for Strategy & Research. Rust for High-Performance Backtesting.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.3-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?style=flat-square" alt="Code style">
</p>

**trade-learn** aims to eliminate the friction between quantitative research (Learn) and backtest execution (Trade). Through a hybrid architecture of "Python logic + Rust native core," it achieves a **110x+** performance leap in multi-asset backtesting while ensuring **100% rigorous alignment** with Backtrader. It reduces large-scale validation time from hours to seconds, providing a lightning-fast iteration experience for index enhancement and machine learning strategies.

Beyond extreme performance, the project focuses on scientific rigor. To combat "pseudo-correlation" risks common in machine learning, we have deeply integrated **Causal Inference** into the research workflow. By identifying the true causal driving paths of factors, it effectively reduces out-of-sample decay, helping you build highly explainable and robust quantitative systems.

To transform scientific methodology into productivity, trade-learn is more than just a high-performance engine; it is a complete full-lifecycle research pipeline **with built-in JupyterLab and MLflow**. It seamlessly links factor mining, strategy validation, and experiment auditing, ensuring every research decision is fully traceable and allowing researchers to focus on the essence of strategy development.

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## Implementation Path

**trade-learn** adopts a "Layered Design, Dual-Mode Drive" architecture, balancing professional depth with research efficiency.

You can flexibly choose the depth of strategy expression based on your research stage:
- **Engine Mode (Deep Alignment)**: Full support for the Analyzer / Sizer / Signal ecosystem. The underlying Engine is deeply aligned with Backtrader semantics, designed for building complex systems with extreme precision.
- **Lite Mode (Agile Iteration)**: Extreme minimalism with support for direct model weight connection. The built-in `target_weights` interface converts research outputs into backtest decisions with one click, perfect for high-frequency iteration during factor mining.

Furthermore, the project creatively embeds causal discovery into automated pipelines via **CausalSelector**. It automatically connects feature selection, parameter optimization, and backtest auditing, ensuring every selected factor possesses true causal explanatory power rather than mere statistical coincidence.

## Key Highlights

#### ⚡️ High-Performance Core: Extreme Performance Driven by Rust
- **Rust Hybrid Drive**: The core matching engine is powered by Rust, providing **28x** acceleration for single-symbol and **110x+** for multi-asset rebalancing compared to Backtrader.
- **Intelligent Runner Scheduling**: Automatically switches execution modes based on data shape. **Optimized memory layout for index enhancement scenarios** ensures extremely low latency even in large-scale backtests.

#### 🛡️ Rigorous Finance: 100% Backtrader Semantic Alignment
- **Engine-Level Deep Alignment**: Full support for the Analyzer / Sizer / Signal system, ensuring every trade's execution logic is zero-divergence from the Backtrader official results, with high support for self-extending components.
- **Lite Agile Syntax**: A lightweight expression built on the same high-performance Runtime. **Built-in `target_weights` interface** converts ML model weights into decisions with one click.

#### 🧪 Causal Research: Scientific Workflow Beyond Correlation
- **Causal-First Factor Selection**: Integrated with PC / FCI causal discovery algorithms to identify true driving paths, combating "pseudo-correlation" and overfitting from the source.
- **Pipeline Experiment Closure**: Seamlessly couples feature engineering, causal screening, scoring models, portfolio weights, and backtest reports into a reproducible experiment loop.

#### 📦 Modular Platform: Light Core, Scale on Demand
- **Decoupled Architecture**: The default installation includes only the high-performance core, with minimal dependencies, making it easy to integrate into servers or automated trading systems.
- **Elastic Extension**: One-click activation of the integrated environment (**JupyterLab + MLflow + AI Assistant**) via `[lab]` or `[all]` extras, enabling "load on demand, run anywhere."

#### 🌍 Global Vision: Multi-Standard Indicators and Modern Ecosystem
- **Dual-Market Compatibility**: Explicit support for TDX (A-share) / TradingView (Global) indicator standards, deeply compatible with TA-Lib and pandas-ta.
- **Modern Tools**: Out-of-the-box HTML interactive reports, MLflow experiment tracking, and deep integration with JupyterLab / MCP.

## Causal Research: Beyond the "Pseudo-Correlation" Trap

Most quantitative research stops at **Statistical Correlation**, which often leads to factors performing excellently in backtests but failing rapidly in live trading (overfitting). trade-learn identifies the true drivers behind returns through its built-in **Causal Discovery** mechanism:

- **Causal Feature Selection**: Using `CausalSelector` combined with PC / FCI algorithms, it strips away pseudo-correlated factors caused by "common observations," retaining only features with direct driving capacity for returns.
- **Resistance to Out-of-Sample Decay**: Alpha factors identified via causal graphs possess stronger survival capabilities during market style shifts, effectively reducing the performance gap from research to production.
- **Industrial-Grade Integration**: Deeply integrated with the `causal-learn` ecosystem, making cutting-edge causal inference as smooth as calling `corr()`, significantly lowering the barrier for academic algorithms.

## Who is it for?

*   **Agile Developers & Idea Validation**:
    Tired of heavy configurations and wanting to transform ideas into backtest reports within a few lines of code, enjoying a lightweight experience similar to backtesting.py.
*   **Index Enhancement & Portfolio Management**:
    Facing large-scale backtesting with 1000+ symbols, utilizing the Rust Panel Runner for second-level rebalancing simulation, saying goodbye to long waits in traditional frameworks.
*   **Machine Learning & Factor Research**:
    Aiming to integrate feature engineering, **Causal Discovery**, model training (MLflow tracking), and backtesting into a one-stop automated loop.
*   **Backtrader Power Users**:
    Seeking a more modern reporting system, full-link pipelines, and high-performance Rust backtesting while retaining mature event-driven semantics.
*   **Cross-Market & Multi-Strategy Teams**:
    *   **Unified Standards**: Unified indicator standards and reporting systems across A-shares (TDX) and Global markets (TradingView).
    *   **Full System Maintenance**: Unified management of rule-based and model-based strategies, reducing R&D costs caused by toolchain fragmentation.
*   **Causal Inference Explorers**:
    Dedicated to introducing causal graph technology in the factor selection stage to build highly explainable and robust quantitative systems.

## Installation

```bash
pip install trade-learn
```

Get the latest version:

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

Optional extras:

| extra | Usage |
|---|---|
| `[lab]` | JupyterLab / Jupyter AI / MCP / Pygwalker interactive research environment |
| `[mlflow]` | MLflow tracking server and experiment artifact recording |
| `[all]` | Full environment including Lab, MLflow, Riskfolio-Lib, Optuna, DuckDB, etc. |

> **💡 Installation Suggestion**:
> The default installation only includes the core backtest engine. To enable the full-stack research experience with JupyterLab and MLflow, please specify the `[all]` extra:
> ```bash
> pip install "trade-learn[all]"
> ```
> After starting `tradelearn lab` in the project root, you can access the research environment via port `8888` and the MLflow dashboard via port `5050` by default.

## Quick Start

**Lite — The Shortest Path** (for fast validation, teaching, and multi-asset target weights):

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
> **Multi-symbol Logic:** In multi-symbol backtesting, the strategy binds to `self.data` (the primary data source) by default. This means the code above will only execute based on the signals of the first symbol even if multiple symbols are provided. To implement independent parallel trading across multiple assets, iterate through `self.datas` in the strategy's `init` to build indicators for each data source.

**Engine — Backtrader Style** (for complex/portfolio strategies and future paper/live modes):

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
> **Multi-symbol Logic:** In multi-symbol backtesting, the strategy binds to `self.data` (the primary data source) by default. This means the code above will only execute based on the signals of the first symbol even if multiple symbols are provided. To implement independent parallel trading across multiple assets, iterate through `self.datas` in the strategy's `init` to build indicators for each data source.

## Research Pipeline Example

The README only shows the shortest readable version. For complete scripts, see [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) and [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py).

**1. Research: Generate features and split train/test sets**

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

**2. Pipeline: Preprocessing, model scoring, and weight generation**

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

**3. Portfolio: Execute target weights via Lite / Engine**

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

**4. Live-style: Inference using only the current visible window**

Research pipelines are suitable for offline training; for live-like strategy semantics, you can put the model into strategy parameters and use `history_panel()` in `next()`.

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

Full versions:

| Target | Full Script |
|---|---|
| Lite Research + Backtest + Report + MLflow | [`examples/research/index_enhance_lite_pipeline.py`](./examples/research/index_enhance_lite_pipeline.py) |
| Engine Research + Backtest + Report + MLflow | [`examples/research/index_enhance_engine_pipeline.py`](./examples/research/index_enhance_engine_pipeline.py) |
| Lite Live-style current window inference | [`examples/research/index_enhance_lite_live.py`](./examples/research/index_enhance_lite_live.py) |
| Engine Live-style current window inference | [`examples/research/index_enhance_engine_live.py`](./examples/research/index_enhance_engine_live.py) |
| Engine Backtrader style portfolio rebalancing | [`examples/engine/11_target_percent_portfolio.py`](./examples/engine/11_target_percent_portfolio.py) |
| Asset class portfolio strategy | [`examples/engine/12_asset_class_portfolios.py`](./examples/engine/12_asset_class_portfolios.py) |

## Performance & Alignment

The local baseline focuses on two cores: **Result Alignment** and **Throughput Speed vs Backtrader**. See [Benchmarks](./docs/benchmarks.md) for full commands.

#### 1. Single-Symbol High-Frequency Stress Test: SMA Cross (550k Bars)
* **Strategy**: Standard SMA cross. Aims to stress-test Rust's event-driven performance and state maintenance efficiency for long sequences.

| Engine Mode | Time | Throughput (Bars/s) | **Acceleration** | Equity | Trades | Closed | Status |
|---|---|---|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **118,399.33** | 10,299 | 5,149 | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | 118,399.33 | 10,299 | 5,149 | - |

#### 2. Multi-Symbol Large-Scale Index Enhancement: Top-50 Weights (5.04M Bars)
* **Strategy**: 1000-symbol market-wide stock selection. Aims to stress-test Rust's memory layout optimization and concurrent processing for large-scale Panel data.

| Engine | Time | bars/s | **Acceleration** | Alignment |
|---|---:|---:|---:|---:|
| **Lite** | 2.40s | **2,094,237** | **119.1x** | Final Value 4,199,638.26 |
| **Engine** | 4.11s | **1,225,594** | **69.7x** | Final Value 4,199,638.26 |
| Backtrader | 286.53s | 17,589 | 1.0x | Final Value 4,199,638.26 |

## Consistency Commitment

**trade-learn** regards "Baseline Comparison" as a core engineering discipline. We ensure every numerical result is rigorously verified in the following dimensions:

*   **Financial Metrics**: `metrics` (Sharpe, MaxDD, Sortino, etc.) fully aligned with `empyrical`, error controlled within `rtol=1e-10`.
*   **Multi-Source Indicators**:
    *   `tl.pta` aligned with `pandas-ta-classic`: `rtol=1e-10`.
    *   `tl.tdx` aligned with `MyTT`: `rtol=1e-10`.
    *   `tl.tv` aligned with `pyneCore`: `rtol=1e-6`.
*   **Backtest Engine**:
    *   **Decision Layer**: Execution records (**Trades**) zero-divergence from Backtrader official results (matching in time, direction, and position).
    *   **Equity Layer**: Equity curve error `rtol=1e-6`, summary statistics error `rtol=1e-4`.

> [!IMPORTANT]
> We maintain a "zero-tolerance" attitude toward any numerical micro-discrepancy. All deviations are documented with cause analysis. See [Design Notes → Semantic Consistency Audit](docs/internals/consistency.md).

## Full Documentation

*   **Official Online Docs**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **Local Manual**: [`docs/`](./docs/README.md)

| Topic | Entrance |
|---|---|
| First Backtest in 30 Lines | [Quick Start](./docs/quickstart.md) |
| Lite / Engine Usage | [Lite Guide](./docs/guides/lite.md) · [Engine Guide](./docs/guides/engine.md) |
| Architecture & Boundaries | [Architecture](./docs/concepts/architecture.md) |
| Factor / ML / Weight Research Pipeline | [Research Guide](./docs/guides/research.md) |
| Multi-Standard Indicators | [Indicators Guide](./docs/guides/indicators.md) |
| Performance Benchmarks | [Benchmarks](./docs/benchmarks.md) |
| Internals (Contracts / Matching / Portfolio) | [Design Notes](./docs/internals/contracts.md) |
| Full API Reference | [API Reference](./docs/api/reference.md) |

## 🚀 Roadmap

*   **v1.0.x (Current - Stable Release)**
    *   [x] Rust-based Multi-symbol Clocked Runner.
    *   [x] Complete Index Enhancement Pipeline (Research -> Weight -> Backtest).
    *   [x] Deeply integrated MLflow tracking and modern HTML reports.
*   **v1.1.x (Advanced Research)**
    *   [ ] **Enhanced Causal Inference**: More algorithms (GIES, Direct-LiNGAM) for factor explainability.
    *   [ ] **High-Performance Connectors**: Direct connection to DolphinDB and DuckDB.
    *   [ ] **Risk Models**: Barra-style exposure analysis and return decomposition.
*   **v1.2.x (Live & Production)**
    *   [ ] **Live Adapters**: Universal live event interface for brokers (QMT, etc.).
    *   [ ] **Distributed Optimization**: Parallel parameter search via Ray/Optuna.
    *   [ ] **Agent Integration**: Automated research control via MCP.

## Credits

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## Contact

Email: muyes88@gmail.com
