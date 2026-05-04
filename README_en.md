<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="600" />
</p>

<p align="center">
  <strong>Python for Strategy and Research, Rust for Event-Driven Backtesting Core.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v1.0.0-orange.svg" alt="PyPI version">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python versions">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style">
</p>

<p align="center">
  <a href="./README.md">中文版</a> | <a href="./README_ja.md">日本語版</a>
</p>

**trade-learn** is a quantitative framework that integrates **trade** and **learn** into a single pipeline, designed for index enhancement, quant research, machine learning strategies, and event-driven backtesting. Python provides the flexibility for strategy expression, factor research, and model experimentation, while Rust handles the high-frequency backtesting core, including matching, order advancement, and portfolio calculation. From research and backtesting to reporting and experiment tracking, everything revolves around a single reproducible workflow.

The core pain point it addresses is no longer just "how to run a backtest," but how to weave fragmented research segments into a complete strategy lifecycle:

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

## Implementation Path

trade-learn avoids simple feature stacking; instead, it builds a bridge between "professional depth" and "development efficiency." The underlying **Engine** is deeply aligned with Backtrader semantics to ensure logical correctness, while the upper **Lite** layer provides a minimalist Pythonic interface. Both share a high-performance Runtime, ensuring the perfect unification of "Speed" and "Accuracy."

You can define the "thickness" of your strategy based on the development stage:
- **Engine Mode (Deep Research)**: Fully aligned with Backtrader semantics, supporting the complete Analyzer/Sizer/Signal ecosystem. Ideal for building complex, production-grade systems with high precision.
- **Lite Mode (Agile Validation)**: Follows the minimalism of backtesting.py, supporting direct connection to model weights. Perfect for high-frequency iteration and prototype validation during the factor mining stage.

It is not only seamlessly compatible with mainstream indicator libraries like TDX, TA-Lib, and TradingView, but also creatively introduces **Causal Inference** into factor research. Through the built-in `CausalSelector`, the project organically links feature selection, parameter optimization, and backtest reports, presenting you with a closed-loop, transparent, and efficient quantitative research pipeline.

## Key Highlights

#### ⚡️ High-Performance Core: Extreme Performance Driven by Rust
- **Rust Hybrid Power**: The matching engine and core calculations are powered by Rust, providing **28x** acceleration for single-symbol and **110x+** for multi-asset rebalancing compared to Backtrader.
- **Automatic Runner Scheduling**: Automatically selects between "Single-Stream Bar-by-Bar" or "Panel Batch" based on data shape. **Optimized memory layout for index enhancement scenarios**, allowing developers to focus solely on `next()` logic.

#### 🛡️ Rigorous Finance: 100% Backtrader Semantic Alignment
- **Engine-Level Alignment**: Full support for the Analyzer / Sizer / Signal system, ensuring zero logical divergence between Trade logs and the Backtrader Oracle.
- **Lite Minimalist Expression**: A lightweight syntax built on the same Runtime. **Built-in `target_weights` interface** to convert machine learning model outputs into backtesting decisions with one click.

#### 🧪 Causal Research: Scientific Workflow Beyond Correlation
- **Causal-First Feature Selection**: Built-in causal discovery algorithms like PC / FCI identify the true driving paths of factors, combating "pseudo-correlation" and overfitting from the source.
- **Pipeline Full-Link Workflow**: Seamlessly couples feature engineering, causal screening, scoring models, portfolio weights, and backtest reports into a reproducible experiment loop.

#### 🌍 Global Vision: Multi-Standard Indicators and Modern Ecosystem
- **Dual-Market Standards**: Explicit support for TDX (A-share) / TradingView (Global) indicator standards, deeply compatible with TA-Lib and pandas-ta.
- **Modern Tools**: Out-of-the-box HTML interactive reports, MLflow experiment tracking, and deep integration with JupyterLab / MCP.

## Causal Research: Beyond the "Pseudo-Correlation" Trap

Most quantitative research stops at **Correlation**, which often leads to factors performing excellently in backtests but failing rapidly in live trading (overfitting). trade-learn identifies the true drivers behind returns through its built-in **Causal Discovery** mechanism:

- **Causal Feature Selection**: Using `CausalSelector` combined with PC / FCI algorithms, it strips away pseudo-correlated factors caused by "common observations," retaining only features with direct driving capacity for returns.
- **Resistance to Out-of-Sample Decay**: Alpha factors identified via causal graphs possess stronger survival capabilities during market style shifts, effectively reducing the performance gap from research to production.
- **Industrial-Grade Integration**: Deeply integrated with the `causal-learn` ecosystem, making cutting-edge causal inference as smooth as calling `corr()`, significantly lowering the barrier for academic algorithms.

## Who is it for?

*   **⚡️ Agile Developers & Idea Validation**
    Tired of heavy configurations and wanting to transform ideas into backtest reports within a few lines of code, enjoying a lightweight experience similar to backtesting.py.
*   **📈 Index Enhancement & Portfolio Management**
    Facing large-scale backtesting with 1000+ symbols, utilizing the Rust Panel Runner for second-level rebalancing simulation, saying goodbye to long waits in traditional frameworks.
*   **🧠 Machine Learning & Factor Research**
    Aiming to integrate feature engineering, **Causal Discovery**, model training (MLflow tracking), and backtesting into a one-stop automated loop.
*   **🛠️ Backtrader Power Users**
    Seeking a more modern reporting system, full-link pipelines, and high-performance Rust backtesting while retaining mature event-driven semantics.
*   **🌐 Cross-Market & Multi-Strategy Teams**
    Unified indicator standards and reporting systems across A-shares (TDX) and Global markets (TradingView).
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

provider = TradingViewProvider(n_bars=500)
bars = provider.history_ohlc("NASDAQ:AAPL", start="2023-01-01", end="2024-01-01")

bt = tl.Backtest(bars, LiteSmaCross, cash=100_000, commission=0.0003, trade_on_close=True)
stats = bt.run()

print(stats.summary)
bt.plot()
bt.report("report.html")
```

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

## Performance & Alignment

The local baseline focuses on two cores: **Result Alignment** and **Throughput Speed vs Backtrader**. See [Benchmarks](./docs/benchmarks.md) for full commands.

#### 1. Single-Symbol High-Frequency Stress Test: SMA Cross (550k Bars)
| Engine | Time | bars/s | **Acceleration** | Alignment |
|---|---:|---:|---:|---:|
| **Lite** | 1.32s | **414,990** | **27.9x** | Final Value 118,399.33 |
| **Engine** | 3.37s | **162,883** | **11.0x** | Final Value 118,399.33 |
| Backtrader | 37.02s | 14,854 | 1.0x | Final Value 118,399.33 |

#### 2. Multi-Symbol Large-Scale Index Enhancement: Top-50 Weights (5.04M Bars)
| Engine | Time | bars/s | **Acceleration** | Alignment |
|---|---:|---:|---:|---:|
| **Lite** | 2.40s | **2,094,237** | **119.1x** | Final Value 4,199,638.26 |
| **Engine** | 4.11s | **1,225,594** | **69.7x** | Final Value 4,199,638.26 |
| Backtrader | 286.53s | 17,589 | 1.0x | Final Value 4,199,638.26 |

## Documentation

*   **Official Online Docs**: [**https://muuyesen.github.io/trade-learn/**](https://muuyesen.github.io/trade-learn/)
*   **Local Manual**: [`docs/`](./docs/README.md)

## Roadmap

*   **v1.0.x (Current - Stable Release)**: Rust clocked runner, index enhancement pipeline, MLflow integration.
*   **v1.1.x (Advanced Research)**: Enhanced causal inference, high-performance connectors (DolphinDB/DuckDB), Risk models (Barra style).
*   **v1.2.x (Live & Production)**: Live trading adapters (QMT, etc.), Distributed optimization (Ray/Optuna), Agent integration (MCP).

## License

Apache-2.0. See [`NOTICE`](./NOTICE) for upstream attributions.

## Contact

Email: muyes88@gmail.com
