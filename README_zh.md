<p align="center">
  <img src="docs/tradelearn-logo.png" alt="trade-learn logo" width="600" />
</p>

<p align="center">
  <strong>Python 写策略与投研流程，Rust 扛事件驱动回测内核。</strong>
</p>

<p align="center">
  <a href="./README.md">English version</a>
</p>

**trade-learn** 是一个面向量化研究、机器学习策略和事件驱动回测的 Python / Rust 框架。Python 保留策略表达、因子研究和模型实验的灵活性，Rust 承担撮合、订单推进、portfolio 计算这类高频回测内核——研究、回测、报告与实验追踪连成一条可复现的工作流。

它想解决的不是"怎么跑一段回测"，而是把一条完整的策略研发链路接起来：

<p align="center">
  <img src="docs/research-flow.png" alt="trade-learn research flow" width="100%" />
</p>

你可以像写 Backtrader 一样写专业策略，也可以用 Lite API 快速验证一个想法；可以接入 TDX、TA-Lib、TradingView、pandas-ta-classic 等指标生态，也可以把因子分析、因果特征筛选、Optuna 参数优化、组合权重、回测报告和实验记录放进同一条工作流。

## 核心亮点

- **Lite 是推荐起点**：更短的写法，适合快速验证、教学、1.x 风格迁移和多资产目标权重；不是另一套撮合逻辑，是同一 runtime 的轻量语法。
- **Backtrader 风格 Engine**：成熟事件驱动模型，适合复杂策略、组合策略、Analyzer / Sizer / Signal 和未来 paper / live adapter。
- **Rust single / multi-data runner**：单标的走 single-data runner，多标的 panel 自动切到 multi-data clock runner，用户仍然只写 `next()`。
- **性能基线透明**：本机基线（Backtrader 为 1.0x）：
  - 55 万 bar 单标的：Lite 约 **27.9x**、Engine 约 **11.0x**
  - 1000 标的 20 年目标权重：Lite 约 **119.1x**、Engine 约 **69.7x**
- **双市场指标生态**：A 股偏 TDX / MyTT 口径（`tl.tdx` / `bt.tdx`），海外偏 TradingView（`tl.tv` / `bt.tv`）+ pandas-ta-classic / TA-Lib（`tl.pta` / `tl.talib`），口径**显式选择**。
- **机器学习与因果筛选**：`FeatureSet` / `Pipeline` / `CausalSelector` / `ResearchRun` / `Allocator` 把训练 / 测试 / 预处理 / 评分 / 权重 / 回测连成一条链。
- **因子与报告**：alphalens / pyfolio 风格分析，HTML 报告、交互式 plot、CSV / XLSX artifacts。
- **MLflow / JupyterLab / MCP**：实验追踪、交互研究、自动化工具集成开箱即用。

## 适合谁

- 已经会 Backtrader，但希望有更现代的报告、研究流水线和 Rust 回测内核
- 在做因子研究，希望从 alphalens 风格分析自然走到事件驱动回测
- 在做机器学习策略，希望把 train / test、预处理、因果筛选、评分、权重、回测和 MLflow 记录连起来
- 同时覆盖 A 股和海外市场，不想让指标口径、数据形态和报告体系割裂
- 同时维护规则策略和模型策略，不想让两套策略使用完全不同的数据、报告和实验体系

## 一致性承诺

trade-learn 把"对照基线"当作工程纪律：

- `metrics`（sharpe / max_dd / sortino / ...）对 empyrical：`rtol=1e-10`
- `tl.pta` / `bt.pta` 对 pandas-ta-classic：`rtol=1e-10`
- `tl.tdx` / `bt.tdx` 对 MyTT：`rtol=1e-10`
- `tl.tv` / `bt.tv` 对 pyneCore / TradingView：`rtol=1e-6`
- 回测 **trades**（决策层）对 Backtrader oracle：**0 差异**（时间 / 方向 / size）
- 回测 equity 对 oracle：`rtol=1e-6`，summary：`rtol=1e-4`，每条差异都登记在案

详见 [Design Notes → 与外部库的语义一致性](docs/internals/consistency.md)。

## 安装

```bash
pip install trade-learn
```

获取最新版本：

```bash
pip install git+https://github.com/MuuYesen/trade-learn.git@master
```

可选 extras：`[lab]`（JupyterLab）、`[live-qmt]`（仅 Windows 实盘 broker，1.1 起提供）。

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

## 完整文档

完整技术手册（mkdocs 站点）：[`docs/`](./docs/README.md)

| 主题 | 入口 |
|---|---|
| 30 行走通第一个回测 | [快速开始](./docs/quickstart.md) |
| Lite / Engine 用法 | [Lite Guide](./docs/guides/lite.md) · [Engine Guide](./docs/guides/engine.md) |
| 架构与边界 | [架构](./docs/concepts/architecture.md) |
| 因子 / ML / 权重研究流水线 | [Research Guide](./docs/guides/research.md) |
| 双口径指标（`tl.talib` / `tl.pta` / `tl.tdx` / `tl.tv`） | [Indicators Guide](./docs/guides/indicators.md) |
| 性能基线 | [Benchmarks](./docs/benchmarks.md) |
| 内核（契约 / 撮合 / portfolio / 事件循环） | [Design Notes](./docs/internals/contracts.md) |
| 完整 API | [API Reference](./docs/api/reference.md) |

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
