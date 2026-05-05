<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="550" />
</p>

<p align="center">
  <a href="https://muuyesen.github.io/trade-learn/"><b>Documentation</b></a> |
  <a href="./CHANGELOG.md"><b>Changelog</b></a> |
  <a href="./README.md"><b>中文简体</b></a> |
  <a href="./README_ja.md"><b>日本語</b></a>
</p>

<p align="center">
  <strong>Python for Strategy & Research, Rust for Event-Driven Backtest Engine.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v0.2.4-orange?style=flat-square" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Changelog-v0.2.4-blue?style=flat-square" alt="Changelog">
</p>

**trade-learn** aims to eliminate the long-standing friction between quantitative research ("Learn") and backtest execution ("Trade"). By adopting a hybrid architecture of "Python for strategy logic + Rust for native backtest core," it achieves a **110x+ performance leap** in multi-asset backtesting while ensuring **100% rigorous semantic alignment** with Backtrader. It compresses large-scale strategy validation from hours to seconds, providing truly iterative research efficiency for index enhancement and machine learning strategies.

Beyond high performance, **trade-learn** provides a complete research infrastructure. With built-in **JupyterLab** and **MLflow**, it seamlessly links factor mining, strategy validation, and experiment auditing into a **reproducible, traceable, and auditable** full-lifecycle research pipeline. This elevates the research process from "result-oriented" to a systematically managed engineering workflow, allowing researchers to focus on the core strategy logic.

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

**From Extreme Efficiency to Scientific Decision-Making**: Building on the efficiency gains of the research pipeline, **trade-learn** further addresses the core "scientific rigor" of quantitative research. To combat the "pseudo-correlation" risks common in machine learning strategies, we have deeply integrated **Causal Inference** into the research workflow. By identifying true causal driving paths, it reduces out-of-sample decay risks and helps build highly explainable and robust quantitative strategy systems.

## Implementation Path

**trade-learn** rejects the simple stacking of features. Instead, it balances professional depth with research efficiency through a unique "Dual-Mode, Dual-Core" design. The **Engine** layer strictly aligns with Backtrader semantics for logic correctness, while the **Lite** layer provides a minimal Pythonic interface for rapid iteration.

You can define the depth of your strategy based on the research stage:
- **Engine Mode (Deep Research)**: Fully aligns with Backtrader semantics, supporting the complete Analyzer/Sizer/Signal ecosystem. Ideal for building complex, production-grade systems with precise logic.
- **Lite Mode (Agile Validation)**: Follows the minimalism of `backtesting.py`, supporting direct connection to model weights. Perfect for high-frequency iteration and prototype validation during the factor mining stage.

In terms of ecosystem, **trade-learn** provides comprehensive indicator support, compatible with TA-Lib, Pandas-TA-Classic, TDX, and TradingView, while allowing flexible expansion of custom indicators and data sources.

## Core Highlights

#### ⚡️ High-Performance Core: Rust-Driven Velocity
- **Rust Hybrid Power**: The matching engine and core calculations are powered by Rust, providing **28x** speedup for single assets and **110x+** for multi-asset rebalancing compared to Backtrader.
- **Automatic Runner Scheduling**: Automatically selects between "Single-Stream Bar-by-Bar" or "Batch Panel" processing based on data shape. Optimized memory layout for **Index Enhancement** scenarios.

#### 🛡️ Rigorous Finance: 100% Backtrader Alignment
- **Engine-Level Alignment**: Full support for the Analyzer/Sizer/Signal system, ensuring zero logical divergence from the Backtrader Oracle.
- **Lite Minimalist Expression**: Lightweight syntax built on the same runtime. Features a built-in `target_weights` interface to convert ML model outputs into backtest decisions instantly.

#### 🧪 Causal Research: Scientific Workflow Beyond Correlation
- **Causal-First Feature Selection**: Built-in causal discovery algorithms like PC/FCI to identify true causal paths and combat "pseudo-correlation" and overfitting.
- **Full-Link Pipeline**: Seamlessly couples feature engineering, causal screening, scoring models, portfolio weights, and backtest reports into a reproducible experimental loop.

#### 📦 Modular Platform: Lightweight Core, On-Demand Expansion
- **Decoupled Core**: The default installation includes only the high-performance backtest kernel with minimal dependencies, making it easy to integrate into servers or automated trading systems.
- **Elastic Expansion**: One-click activation of the integrated research environment (**JupyterLab + MLflow + AI Assistant**) via `[lab]` or `[all]` extras.

#### 🌍 Global Vision: Multi-Standard Indicators & Modern Ecosystem
- **Dual-Market Standards**: Explicit support for TDX (China) and TradingView (International) indicator standards, with deep compatibility for TA-Lib and Pandas-TA-Classic.
- **Modern Tools**: Out-of-the-box HTML interactive reports, MLflow experiment tracking, and deep JupyterLab/MCP integration.

## Causal Quant: Bridging the "Pseudo-Correlation" Trap

Most quantitative research stops at **Correlation**, which often leads to factors performing well in backtests but failing rapidly in live trading (overfitting). trade-learn helps you identify the true drivers of returns through its built-in **Causal Discovery** mechanism:

- **Causal Feature Selection**: Use `CausalSelector` with PC/FCI algorithms to strip away pseudo-correlated factors caused by "common observations," keeping only features with direct driving capability for returns.
- **Resisting Out-of-Sample Decay**: Alpha factors identified via causal graphs are more resilient to market regime shifts, effectively reducing the performance gap between research and live trading.
- **Industrial Integration**: Deeply integrated with the `causal-learn` ecosystem, making advanced causal inference as seamless as calling `corr()`.

## Who is it for?

*   **Agile Developers & Prototypers**: Convert ideas into backtest reports in just a few lines of code, enjoying a `backtesting.py`-like lightweight experience.
*   **Index Enhancement & Portfolio Managers**: Simulate rebalancing for 1000+ assets in seconds using the Rust Panel Runner.
*   **ML & Factor Researchers**: A one-stop automated loop from feature engineering and **Causal Discovery** to MLflow-tracked model training and backtesting.
*   **Backtrader Power Users**: Modernize your reporting and speed up your research while retaining the mature event-driven semantics you trust.
*   **Cross-Market Teams**: Maintain consistent indicator standards and reporting systems across A-shares (TDX) and International markets (TradingView).

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
| `[lab]` | JupyterLab / Jupyter AI / MCP / Pygwalker environment |
| `[mlflow]` | MLflow tracking server and artifact logging |
| `[all]` | Full environment (Lab, MLflow, Riskfolio-Lib, Optuna, DuckDB, etc.) |

> **💡 Installation Tip**:
> The default install includes only the core engine. For the full research experience, use `[all]`:
> ```bash
> pip install "trade-learn[all]"
> ```
> Launch with `tradelearn lab`. Access JupyterLab at port `8888` and MLflow at `5050`.

## Quick Start

**Lite — The Shortest Path** (Ideal for rapid validation, teaching, and target-weight portfolios):

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

**Engine — Backtrader Style** (Ideal for complex portfolios and future paper/live modes):

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

## Alignment & Performance

Our benchmarks focus on two core metrics: **Result Parity** and **Throughput speedup** compared to Backtrader.

#### 1. Single-Asset High-Frequency: SMA Cross (550k Bars)
| Engine Mode | Time | Throughput (Bars/s) | **Speedup** | Status |
|---|---|---|---|---|
| **Tradelearn Lite** | **1.32s** | **414,990** | **27.9x** | **EXACT** |
| **Tradelearn Engine** | **3.37s** | **162,883** | **11.0x** | **EXACT** |
| Backtrader (Oracle) | 37.02s | 14,854 | 1.0x | - |

#### 2. Large-Scale Index Enhance: Top-50 Target Weights (5.04M Bars)
| Engine Mode | Time | Throughput (Bars/s) | **Speedup** |
|---|---|---|---|
| **Tradelearn Lite** | **2.40s** | **2,094,237** | **119.1x** |
| **Tradelearn Engine** | **4.11s** | **1,225,594** | **69.7x** |
| Backtrader (Oracle) | 286.53s | 17,589 | 1.0x |

## Parity Commitment

**trade-learn** treats "Benchmark Parity" as a core engineering discipline:
- **Financial Metrics**: Sharpe, MaxDD, etc., align with `empyrical` at `rtol=1e-10`.
- **Indicators**: `tl.pta` (standard) and `tl.tdx` (China standard) align at `rtol=1e-10`.
- **Engine Parity**: Trades align with the Backtrader Oracle with **0 difference**.

## 🚀 Roadmap

Based on the engineering plan in [PROJECT.md](./design/PROJECT.md), we have divided the evolution into five core dimensions:

#### 🏗️ v1.x (Backtest Engine & Infrastructure)
- [x] **Rust Hybrid Kernel**: Clocked Multi-Data Runner, **110x+** speedup.
- [x] **Backtrader Parity**: 100% logic consistency, shared runtime via `bt.Strategy`.
- [x] **Index Enhance Pipeline**: End-to-end `Data → Factor → Score → Weights`.
- [x] **Automated Audit**: Deep MLflow integration for code snapshots, params, and reports.
- [x] **High-Perf Backend**: **DuckDB native connector landed**, supports sub-second reading of billions of bars.
- [ ] **Risk Models**: Support for Barra-style risk exposure analysis and attribution.

#### 🧪 v1.x+ (Scientific Research)
- [x] **Causal Discovery Base**: Integrated `CausalSelector` (PC/FCI) to identify true drivers.
- [ ] **Algorithm Expansion**: Integrate GIES, Direct-LiNGAM for enhanced explainability.
- [ ] **Causal Loop**: Closed-loop integration of causal analysis with parameter optimization.

#### 🤖 v1.x+ (Agent & AI Capabilities)
- [x] **MCP Knowledge Gateway**: **MCP Server is live**, enabling structured API understanding for LLMs.
- [ ] **Agentic Diagnosis**: LLM-driven analysis of backtest results to identify loss drivers and suggest optimizations.
- [ ] **LLM Factor Interpreter**: Translate causal discovery results into intuitive financial logic.

#### ⚙️ v1.x+ (Engineering & ML Lifecycle)
- [x] **Model Registry**: **MLflow-based Model Registry** for full-lifecycle versioning and tracking.
- [ ] **Distributed Tuning**: Scalable parameter search using Ray / Optuna.

#### 🌍 v2.x (Live Trading & Ecosystem)
- [x] **Universal Event Link**: `EventRunner` semantics for 100% code reuse between backtest and live trading.
- [ ] **Live Connectivity**: Integration with `QMT`, `IBKR`, and other brokers for the "last mile."
- [ ] **Agentic Quant Platform**: Evolving into a semantic-driven automation base for quantitative research.

## Disclaimer

This project is for academic research and technical exchange only and does not constitute any investment advice. Quantitative trading involves high risk; past performance is not indicative of future results. The developers are not responsible for any financial losses incurred through the use of this project. Invest at your own risk.

## Acknowledgements

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [DolphinDB](https://github.com/dolphindb) · [mpquant](https://github.com/mpquant)

## Contact

WeChat: 知守溪的收纳屋 · Email: muyes88@gmail.com
