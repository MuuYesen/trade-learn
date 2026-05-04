<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="600" />
</p>

<p align="center">
  <strong>Python for strategy and research, Rust for the event-driven backtest core.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v1.0.0-orange.svg" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style">
</p>

<p align="center">
  <a href="./README.md">中文主页</a>
</p>

**trade-learn** is a high-performance quantitative framework designed to bridge the gap between **Trading** and **Learning**. It is built for Index Enhancement, quantitative research, machine-learning strategies, and event-driven backtesting. By leveraging **Python** for flexible strategy expression and **Rust** for a high-performance matching kernel (handling order processing and portfolio accounting), it provides a unified and reproducible workflow from factor discovery to experiment tracking.

It focuses not just on "running a backtest," but on connecting the entire strategy research lifecycle:

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## Implementation Path

trade-learn is not just a collection of features; it is a bridge from "Professional Depth" to "Research Efficiency":
- **Engine Layer**: Deeply aligned with Backtrader semantics to ensure industrial-grade correctness.
- **Lite Layer**: Provides a minimal, Pythonic interface for rapid prototyping.
Both layers share the same high-performance Rust Runtime, ensuring the perfect unification of **Accuracy** and **Speed**.

You can define the "depth" of your strategy freely:
- **Engine Mode (Backtrader-style)**: Offers atomic control and full lifecycle events, ideal for building complex, production-ready systems.
- **Lite Mode (Backtesting-like)**: Uses a highly encapsulated, minimal syntax supporting direct target-weight rebalancing, perfect for rapid factor validation.

Beyond mainstream indicators like TDX, TA-Lib, and TradingView, it creatively introduces **Causal Inference** into factor research. Through the built-in `CausalSelector`, it connects feature engineering, causal discovery, and backtesting into a transparent, scientific pipeline.

## Core Highlights

#### ⚡️ High-Performance Kernel: Rust-Powered Engine
- **Rust Hybrid Power**: Core matching and accounting are handled by Rust, providing **28x** speedup for single assets and **110x+** for multi-asset portfolios compared to Backtrader.
- **Automated Dispatching**: Automatically selects between "Single-Bar" and "Panel" runners based on data shape. **Optimized memory layout for Index Enhancement**, allowing researchers to focus solely on `next()` logic.

#### 🛡️ Rigorous Finance: 100% Backtrader Parity
- **Semantic Consistency**: Fully supports Analyzer / Sizer / Signal ecosystems, ensuring trades are identical to the Backtrader "Oracle" baseline.
- **Lite Simplicity**: Built on the same runtime, featuring the **`target_weights` API** to turn ML model outputs into backtest decisions with one line of code.

#### 🧪 Causal Research: Beyond "Spurious Correlation"
- **Causal-First Selection**: Integrated PC / FCI algorithms to identify true driving paths of factors, combating overfitting and "spurious correlations" at the source.
- **Integrated Pipeline**: Couples feature engineering, causal discovery, and backtesting into a single, reproducible experiment loop.

#### 🌍 Global Ecosystem: Modern Tooling
- **Dual Convention**: Explicit support for TDX (A-share) and TradingView (Overseas) indicator conventions, compatible with TA-Lib and pandas-ta.
- **Modern Stack**: Interactive HTML reports, MLflow experiment tracking, and deep integration with JupyterLab / MCP.

## Causal Research: Crossing the "Spurious" Trap

Most quantitative research stops at **Correlation**, which often leads to "backtest-glory but live-failure" (overfitting). trade-learn uses **Causal Discovery** to help you identify the true drivers of returns:

- **Causal Feature Selection**: Uses `CausalSelector` to strip away "common-observer" bias, keeping only features with direct causal paths to returns.
- **Robustness Against Decay**: Alpha factors identified via causal graphs show stronger survival rates during market regime shifts.
- **Industrial Integration**: Deeply integrated with the `causal-learn` ecosystem, making advanced causal inference as simple as calling `.corr()`.

## Who is it for?

*   **📈 Index Enhancement & Portfolio Management**
    Teams handling 1000+ symbols who need second-level rebalancing simulations and want to escape the long wait times of traditional frameworks.
*   **⚡️ Agile Developers & Prototypers**
    Researchers who want to turn an idea into a report in a few lines of code, enjoying a `backtesting.py`-like experience without sacrificing rigor.
*   **🧠 Machine Learning & Factor Researchers**
    Teams wanting to bridge feature engineering, **causal discovery**, and model training (MLflow) into a unified, automated backtesting loop.
*   **🛠️ Backtrader Veterans**
    Users who value mature event-driven semantics but seek modern reporting, full-stack pipelines, and a high-performance Rust kernel.
*   **🔍 Causal Inference Explorers**
    Researchers dedicated to using Causal Graphs in factor selection to build highly interpretable and robust quantitative systems.
*   **🌐 Cross-Market & Multi-Strategy Teams**
    - **Unified Market Specs**: Support for both A-share (TDX) and Overseas (TradingView) conventions in a single system.
    - **Total System Maintenance**: Manage both rule-based and ML-based strategies without fragmented data or reporting pipelines.

## Performance & Alignment

Our baseline focuses on two cores: **Numerical Parity** and **Throughput**. For full replication, see [Benchmarks](./docs/benchmarks.md).

#### 1. Single Asset Pressure Test: SMA Crossover (550k Bars)
*   **Principle**: Standard SMA 10/20 crossover strategy. Designed to test the event-driven efficiency and state maintenance of the Rust kernel under long single-stream sequences.

| Engine | Latency | bars/s | **Speedup** | Alignment |
|---|---:|---:|---:|---:|
| **Lite** | 1.32s | **414,990** | **27.9x** | Final Value 118,399.33 |
| **Engine** | 3.37s | **162,883** | **11.0x** | Final Value 118,399.33 |
| Backtrader | 37.02s | 14,854 | 1.0x | Final Value 118,399.33 |

#### 2. Multi-Asset Portfolio: Top-50 Target Weights (5.04M Bars)
*   **Principle**: Simulating 1000 symbols with a Top-50 target-weight rebalance. Designed to test Rust's memory layout optimization and concurrent processing for large-scale Panel data.

| Engine | Latency | bars/s | **Speedup** | Alignment |
|---|---:|---:|---:|---:|
| **Lite** | 2.40s | **2,094,237** | **119.1x** | Final Value 4,199,638.26 |
| **Engine** | 4.11s | **1,225,594** | **69.7x** | Final Value 4,199,638.26 |
| Backtrader | 286.53s | 17,589 | 1.0x | Final Value 4,199,638.26 |

## Consistency Commitment

**trade-learn** treats "Baseline Alignment" as its highest engineering standard, ensuring every calculation withstands industrial scrutiny:

*   **Financial Metrics**: `metrics` (Sharpe, MaxDD, etc.) aligned with `empyrical` at `rtol=1e-10`.
*   **Multi-Source Indicators**:
    *   `tl.pta` aligned with `pandas-ta-classic` at `rtol=1e-10`.
    *   `tl.tdx` aligned with `MyTT` at `rtol=1e-10`.
    *   `tl.tv` aligned with `pyneCore` (TradingView convention) at `rtol=1e-6`.
*   **Backtest Engine**:
    *   **Decision Layer**: **0 difference** in Trades (time, side, size) compared to the Backtrader Oracle implementation.
    *   **Numerical Layer**: Equity curve at `rtol=1e-6`, summary stats at `rtol=1e-4`.

> [!IMPORTANT]
> We maintain zero tolerance for numerical discrepancies. Every minor deviation is documented with a root-cause analysis. See [Design Notes → Consistency Audit](docs/internals/consistency.md).

## Installation

```bash
pip install trade-learn
```

For the latest features:
```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

## Documentation

*   **Online Documentation**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **Local Handbook**: [`docs/`](./docs/README.md)

| Topic | Link |
|---|---|
| 30-line Quickstart | [Quickstart](./docs/quickstart.md) |
| Lite / Engine Usage | [Lite Guide](./docs/guides/lite.md) · [Engine Guide](./docs/guides/engine.md) |
| Architecture & Concepts | [Architecture](./docs/concepts/architecture.md) |
| Research Pipeline | [Research Guide](./docs/guides/research.md) |
| Indicator Ecosystem | [Indicators Guide](./docs/guides/indicators.md) |
| Benchmarks | [Benchmarks](./docs/benchmarks.md) |
| Internals (Contracts/Matching) | [Design Notes](./docs/internals/contracts.md) |
| API Reference | [API Reference](./docs/api/reference.md) |

## 🚀 Roadmap

*   **v1.0.x (Stable Release - Current)**
    *   [x] Rust-powered Clocked Multi-Data Runner (Stage 13).
    *   [x] End-to-end Index Enhancement Pipeline (Research -> Weight -> Backtest).
    *   [x] Deep integration with MLflow experiment tracking & modern HTML reports.
*   **v1.1.x (Advanced Research)**
    *   [ ] **Enhanced Causal Inference**: Integration of more causal graph algorithms (e.g., GIES, Direct-LiNGAM) for better factor interpretability.
    *   [ ] **High-Performance Connectors**: Native connectors for DolphinDB and DuckDB, enabling second-level reads for billion-bar datasets.
    *   [ ] **Risk Modeling**: Introduction of Barra-style risk exposure analysis and active return attribution.
*   **v1.2.x (Live & Production)**
    *   [ ] **Live Trading Adapters**: Open universal live event interfaces with support for broker terminals like QMT.
    *   [ ] **Distributed Optimization**: Multi-machine parallel parameter search based on Ray/Optuna.
    *   [ ] **Agent Integration**: Automated control of quantitative pipelines via MCP protocol and LLM agents.

## License

Apache-2.0. See [`NOTICE`](./NOTICE) for upstream attributions.

## Acknowledgements

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## Contact

WeChat: 知守溪的收纳屋 · Email: muyes88@gmail.com
