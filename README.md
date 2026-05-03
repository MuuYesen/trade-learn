<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="600" />
</p>

<p align="center">
  <strong>Python for strategy and research, Rust for the event-driven backtest core.</strong>
</p>

<p align="center">
  <a href="./README_zh.md">中文版介绍</a>
</p>

**trade-learn** is a Python / Rust framework for quantitative research, machine-learning strategies, and event-driven backtesting. Python keeps its flexibility for strategy expression, factor research, and model experimentation; Rust handles matching, order processing, and portfolio accounting — so research, backtesting, reporting, and experiment tracking form one reproducible pipeline.

Its goal is not "another backtester," but to connect a complete strategy research loop:

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

You can write a Backtrader-style professional strategy or use the Lite API to validate an idea quickly. You can plug into TDX, TA-Lib, TradingView, and pandas-ta-classic indicator ecosystems, and combine factor analysis, causal feature selection, Optuna parameter search, portfolio weights, backtest reports, and experiment tracking in the same workflow.

## Highlights

- **Lite as the recommended entry point** — short syntax for quick validation, teaching, 1.x-style migration, and multi-asset target weights. Not a separate matching engine — same runtime, lighter syntax.
- **Backtrader-style Engine** — mature event-driven model for complex / portfolio strategies, Analyzer / Sizer / Signal, and future paper / live adapters.
- **Rust single / multi-data runner** — single-data goes through the single-data runner; panel data switches automatically to the multi-data clock runner. Users still only write `next()`.
- **Transparent performance baseline** — locally measured, with Backtrader as the 1.0x baseline:
  - 550k-bar single-asset: Lite ≈ **27.9x**, Engine ≈ **11.0x**
  - 1000-symbol 20-year target-weight portfolio: Lite ≈ **119.1x**, Engine ≈ **69.7x**
- **Dual-market indicator ecosystem** — A-share leans on TDX / MyTT (`tl.tdx` / `bt.tdx`), overseas / general research leans on TradingView (`tl.tv` / `bt.tv`) and pandas-ta-classic / TA-Lib (`tl.pta` / `tl.talib`). Users explicitly choose conventions.
- **ML and causal feature selection** — `FeatureSet`, `Pipeline`, `CausalSelector`, `ResearchRun`, `Allocator` chain train / test, preprocessing, scoring, weights, and backtesting.
- **Factor and reporting** — alphalens / pyfolio style analysis with HTML reports, interactive plots, CSV / XLSX artifacts.
- **MLflow / JupyterLab / MCP** — integrated experiment tracking, interactive research, and LLM tool integration.

## Consistency commitment

trade-learn treats "matching reference baselines" as engineering discipline:

- `metrics` vs empyrical: `rtol=1e-10`
- `tl.pta` / `bt.pta` vs pandas-ta-classic: `rtol=1e-10`
- `tl.tdx` / `bt.tdx` vs MyTT: `rtol=1e-10`
- `tl.tv` / `bt.tv` vs pyneCore / TradingView: `rtol=1e-6`
- Backtest **trades** (decision layer) vs Backtrader oracle: **0 difference** in time / side / size
- Backtest equity vs oracle: `rtol=1e-6` · summary: `rtol=1e-4` (every diff documented)

See [Design Notes → Consistency](docs/internals/consistency.md).

## Install

```bash
pip install trade-learn
```

For the latest:

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

Optional extras: `[lab]` (JupyterLab), `[live-qmt]` (Windows-only live broker, available from 1.1).

## Quickstart

**Lite — shortest path** (good for prototyping, teaching, multi-asset target weights):

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

**Engine — Backtrader-style** (good for complex / portfolio strategies and future paper / live mode):

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

## Documentation

Full technical handbook (mkdocs site): [`docs/`](./docs/README.md)

| Topic | Read |
|---|---|
| 30-line walkthrough | [Quickstart](./docs/quickstart.md) |
| Lite / Engine usage | [Lite Guide](./docs/guides/lite.md) · [Engine Guide](./docs/guides/engine.md) |
| Architecture & boundaries | [Architecture](./docs/concepts/architecture.md) |
| Research pipeline (factor / ML / weights) | [Research Guide](./docs/guides/research.md) |
| Indicators (`tl.talib` / `tl.pta` / `tl.tdx` / `tl.tv`) | [Indicators Guide](./docs/guides/indicators.md) |
| Performance baselines | [Benchmarks](./docs/benchmarks.md) |
| Design Notes (contracts / matching / portfolio / event loop) | [Design Notes](./docs/internals/contracts.md) |
| Full API | [API Reference](./docs/api/reference.md) |

To preview the site locally:

```bash
uv run mkdocs serve
```

## License

Apache-2.0. See [`NOTICE`](./NOTICE) for fused upstream attribution: empyrical / alphalens / pyfolio / quantstats / MyTT / pandas-ta-classic / pyneCore / causallearn / DolphinDB. backtesting.py and backtrader are noted as "inspired by" — not copied.

## Acknowledgements

[Quantopian](https://github.com/quantopian) · [Trevor Stephens](https://github.com/trevorstephens) · [PyWhy](https://github.com/py-why) · [dodid](https://github.com/dodid) · [DolphinDB](https://github.com/dolphindb) · [happydasch](https://github.com/happydasch) · [mpquant](https://github.com/mpquant) · [baobao1997](https://github.com/baobao1997)

## Contact

WeChat Official Account: 知守溪的收纳屋 · Email: muyes88@gmail.com
