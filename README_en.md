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
  <img src="https://img.shields.io/badge/pypi-v1.0.0-orange?style=flat-square" alt="PyPI version">
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

## Quick Start

**1. Initialize Research Project**
```bash
# Create project skeleton (includes config, example strategies, and research notebooks)
tradelearn new my_research
cd my_research
```

**2. Start Full-Stack Environment**
```bash
# One-click activation of JupyterLab + MCP Server + MLflow
# Seamlessly integrates experiment tracking with AI assistant capabilities
tradelearn lab
```

## Key Highlights

#### ⚡️ High-Performance Core: Extreme Performance Driven by Rust
- **Rust Hybrid Drive**: The core matching engine is powered by Rust, providing **28x** acceleration for single-symbol and **110x+** for multi-asset rebalancing compared to Backtrader.
- **Intelligent Runner Scheduling**: Automatically switches execution modes based on data shape. Optimized memory layout for index enhancement scenarios ensures extremely low latency even in large-scale backtests.

#### 🛡️ Rigorous Finance: 100% Backtrader Semantic Alignment
- **Engine-Level Deep Alignment**: Full support for the Analyzer / Sizer / Signal system, ensuring every trade's execution logic is zero-divergence from the Backtrader official results.
- **Lite Agile Syntax**: A lightweight expression built on the same high-performance Runtime. Built-in `target_weights` interface converts ML model weights into decisions with one click.

#### 🧪 Causal Research: Scientific Workflow Beyond Correlation
- **Causal-First Factor Selection**: Integrated with PC / FCI causal discovery algorithms to identify true driving paths, combating "pseudo-correlation" and overfitting from the source.
- **Pipeline Experiment Closure**: Seamlessly couples feature engineering, causal screening, and backtest auditing to build a reproducible and auditable professional research workflow.

#### 📦 Modular Platform: Light Core, Scale on Demand
- **Decoupled Architecture**: The default installation includes only the high-performance core, with minimal dependencies, making it easy to integrate into servers or automated trading systems.
- **Elastic Extension**: One-click activation of the integrated environment (**JupyterLab + MLflow + AI Assistant**) via `[lab]` or `[all]` extras, enabling "load on demand, run anywhere."

#### 🌍 Global Vision: Multi-Standard Indicators and Modern Ecosystem
- **Dual-Market Compatibility**: Explicit support for TDX (A-share) / TradingView (Global) indicator standards, deeply compatible with TA-Lib and pandas-ta.
- **Modern Tools**: Out-of-the-box HTML interactive reports, MLflow experiment tracking, and deep integration with JupyterLab / MCP.

## Who is it for?

*   **⚡️ Agile Developers & Idea Validation**
    Tired of heavy configurations and wanting to transform ideas into backtest reports within a few lines of code, enjoying a lightweight experience similar to backtesting.py.
*   **📈 Index Enhancement & Portfolio Management**
    Facing large-scale backtesting with 1000+ symbols, utilizing the Rust Panel Runner for second-level rebalancing simulation, saying goodbye to long waits in traditional frameworks.
*   **🧠 Machine Learning & Factor Research**
    Aiming to integrate feature engineering, **Causal Discovery**, model training (MLflow tracking), and backtesting into a one-stop automated loop.
*   **🛠️ Backtrader Power Users**
    Seeking a more modern reporting system, full-link pipelines, and high-performance Rust backtesting while retaining mature event-driven semantics.
*   **🔍 Causal Inference Explorers**
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
> After starting `tradelearn lab`, you can access the research environment via port `8888` and the MLflow dashboard via port `5050` by default.

## Documentation

*   **Official Online Docs**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **Local Manual**: [`docs/`](./docs/README.md)

## Roadmap

*   **v1.0.x (Current - Stable Release)**: Rust clocked runner, index enhancement pipeline, MLflow integration.
*   **v1.1.x (Advanced Research)**: Enhanced causal inference, high-performance connectors (DolphinDB/DuckDB), Risk models (Barra style).
*   **v1.2.x (Live & Production)**: Live trading adapters (QMT, etc.), Distributed optimization (Ray/Optuna), Agent integration (MCP).

## Contact

Email: muyes88@gmail.com
