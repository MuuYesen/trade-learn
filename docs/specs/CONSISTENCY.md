# CONSISTENCY

一致性保障体系——重构零信任损失的工程纪律。

## 1. 为什么重要

重构内核 = 用户信任的重大变量。即使 trade-learn 无现有用户,也必须保证:

- **新版 vs 1.x 结果可追溯、可对照**
- **新版 vs 原生 empyrical / pyfolio / alphalens 数值一致**
- **`ta.tdx.*` 对得上通达信软件**
- **`ta.tv.*` 对得上 TradingView 图表**

**"重构后结果不一致"的 bug 一旦流出 = 框架信誉崩塌**。

## 2. 分层容忍度

| 层次 | 对照源 | 容忍 | 说明 |
|---|---|---|---|
| metrics 指标函数 | empyrical / pyfolio / alphalens 原版 | `rtol=1e-10` | 纯函数,必须严格一致 |
| `ta.*` | pandas-ta-classic 原版 | `rtol=1e-10` | 薄封装,无应有差异 |
| `ta.tdx.*` | MyTT 原版 | `rtol=1e-10` | 融合 MyTT,须完全一致 |
| `ta.tdx.*`(抽查) | 通达信软件截图 | `rtol=1e-6` | 手工 3–5 指标 |
| `ta.tv.*` | pyneCore / TV 截图 | `rtol=1e-6` | pyneCore 版本差异容忍 |
| Backtest trades(决策层) | Backtrader oracle | **0 差异**(时间/方向/size) | 兼容目标是“一份策略可同时运行并比较”,Backtrader 为 0.1-alpha golden oracle |
| Backtest equity curve | Backtrader oracle | `rtol=1e-6` | 浮点累积允许差 |
| Backtest 最终 stats | Backtrader oracle | `rtol=1e-4` | 差异必须可解释 |
| 1.x reference | 1.x reference | 记录/迁移参考 | 仅作为历史兼容参考,不得替代 Backtrader parity oracle |

## 3. 金标数据集

### 3.1 结构

```
tests/golden/
├── datasets/                 # 行情数据(parquet)
│   ├── tv/
│   │   ├── GOOG_2020-01-01_2024-12-31_1d.parquet
│   │   ├── AAPL_...
│   │   └── ...(5 个外盘)
│   └── tdx/                  # 历史规划残留;0.1-alpha 不再作为 golden gate
│
├── strategies/               # 标准策略代码
│   ├── sma_cross.py          # 单品种均线穿越
│   ├── rsi_oversold.py       # RSI 超卖
│   ├── bollinger_breakout.py
│   ├── macd_cross.py
│   ├── tdx30_kdj.py          # A 股 KDJ
│   ├── supertrend_tv.py      # 外盘 Supertrend
│   ├── pairs_trading.py      # 多资产配对
│   ├── equal_weight.py       # 等权轮动
│   ├── alpha101_ml.py        # ML + Alpha101
│   └── momentum_portfolio.py # 组合动量
│
├── expected/                 # Backtrader oracle 跑出的 expected
│   └── v1.0/
│       ├── sma_cross_GOOG.json       # {stats, trades, equity_curve}
│       ├── ...
│       └── (10 策略 × 5 TV 数据集 = 50 组)
│
├── indicators/               # 指标金标
│   ├── core/
│   │   └── GOOG_rsi_14.parquet        # pandas-ta-classic 输出
│   ├── tdx/
│   │   ├── 000001_MACD.parquet         # MyTT 输出
│   │   └── 000001_MACD_tdx_screenshot.csv  # 通达信软件手工导出
│   └── tv/
│       └── AAPL_supertrend.parquet     # pyneCore 输出
│
└── returns/                  # metrics 测试用 fixture
    └── synthetic_returns_10.parquet   # 10 条典型 returns 序列
```

### 3.2 版本化

```
tests/golden/expected/v1.0/   # 1.0 发版时冻结
tests/golden/expected/v1.1/   # 若 1.1 有 breaking change,独立存一版
```

金标数据变更流程:

1. 在 `MIGRATION.md` 记录变更原因
2. 新建 `vX.Y/` 目录,保留旧版
3. CI 默认比对最新版,历史版作归档

## 4. 生成金标

### 4.1 初始化

```bash
# 阶段 0 Week 2 执行
python scripts/build_golden.py --version backtrader --engine tv --out tests/golden/expected/v1.0/
```

### 4.2 build_golden.py 逻辑

0.1-alpha 的 `expected/v1.0` 必须来自 Backtrader oracle/parity,不能只用当前 tradelearn 引擎自跑结果冻结。

- Backtrader oracle:用原生 `backtrader.Cerebro` 跑兼容策略,输出标准 orders/trades/fills/equity/summary。
- Tradelearn runner:用 `tradelearn.backtest.Cerebro` 跑同一份或等价兼容策略。
- Parity gate:比较 trades 0 差异、equity `rtol=1e-6`、summary/PnL `rtol=1e-4`。
- Compare gate:`scripts/compare_golden.py --engine tv` 负责对已提供的 expected artifacts 执行 TV subset 全量比较,但不负责生成 Backtrader oracle。
- tradelearn 自跑 expected 仅可作为 smoke/regression,用于证明管道可运行;不得作为 parity oracle。
- 外部 Backtrader 示例仓库只能作为策略语义参考;若无明确 license,不得直接复制大段源码进本仓库,应重写最小等价策略。
- 最小 Backtrader parity smoke 应先覆盖 SMA/MACD/KDJ,再扩展到完整 10 策略。

```python
import sys
sys.path.insert(0, "reference/")   # 优先用 1.x

from tradelearn_1x.query import Query
import backtrader as bt

STRATEGIES = [
    ("sma_cross", SmaCross),
    ("rsi_oversold", RsiOversold),
    # ...
]

DATASETS = [
    {"symbol": "GOOG", "exchange": "NASDAQ", "engine": "tv", "start": "2020-01-01", "end": "2024-12-31"},
    # ...
]

for dataset in DATASETS:
    bars = Query.history_ohlc(**dataset)
    bars.to_parquet(
        "tests/golden/datasets/"
        f"{dataset['engine']}/{dataset['symbol']}_{dataset['start']}_{dataset['end']}_1d.parquet"
    )

    for strat_name, StratCls in STRATEGIES:
        stats = run_backtrader_oracle(bars, StratCls)
        dump_expected(f"tests/golden/expected/v1.0/{strat_name}_{sym}.json", stats)
```

## 5. 一致性测试

### 5.1 目录

```
tests/consistency/
├── test_metrics.py             # metrics 对 empyrical/pyfolio
├── test_indicators_core.py     # ta.* 对 pandas-ta-classic
├── test_indicators_tdx.py      # ta.tdx.* 对 MyTT + 通达信抽查
├── test_indicators_tv.py       # ta.tv.* 对 pyneCore
├── test_backtest_rust.py       # Rust 核对 1.x
├── test_factor.py              # factor 对 alphalens
├── test_report.py              # report.summary 对 pyfolio
└── test_e2e_pipeline.py        # 端到端 Pipeline 对 1.x
```

### 5.2 测试模板

```python
# tests/consistency/test_metrics.py
import empyrical as ep
import pytest
from tradelearn.metrics import sharpe, max_drawdown, sortino

@pytest.mark.parametrize("fixture", load_golden_returns())
@pytest.mark.parametrize("func,ep_func,tol", [
    (sharpe, ep.sharpe_ratio, 1e-10),
    (max_drawdown, ep.max_drawdown, 1e-10),
    (sortino, ep.sortino_ratio, 1e-10),
])
def test_metric_matches_empyrical(fixture, func, ep_func, tol):
    ours = func(fixture.returns, periods=252)
    theirs = ep_func(fixture.returns)
    assert np.isclose(ours, theirs, rtol=tol, equal_nan=True)
```

### 5.3 Trades 决策层

```python
# tests/consistency/test_backtest_rust.py
def test_trades_0_diff_vs_backtrader(golden_strategy):
    strat, sym = golden_strategy
    bars = load_golden_dataset(sym)
    expected = load_expected(strat, sym)

    cerebro = Cerebro()
    cerebro.adddata(bars)
    cerebro.addstrategy(strat)
    actual = cerebro.run()

    # 完全一致
    assert len(actual.trades) == len(expected.trades)
    for a, e in zip(actual.trades, expected.trades):
        assert a.entry_time == e.entry_time        # 时间 0 差异
        assert a.side == e.side                     # 方向 0 差异
        assert np.isclose(a.entry_price, e.entry_price, rtol=1e-6)
        assert np.isclose(a.size, e.size, rtol=1e-6)
```

## 6. CI 集成

### 6.1 Required check

```yaml
# .github/workflows/consistency.yml
name: Consistency
on: [pull_request]
jobs:
  consistency:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run pytest tests/consistency/ -v --tb=short
      - run: uv run pytest tests/golden/ -v
      - name: Fail PR on any tolerance breach
        if: failure()
        run: |
          echo "::error::Consistency test failed. Check docs/MIGRATION.md"
          exit 1
```

在 GitHub 仓库 **Branch protection** 设置 `consistency` 为 required。

### 6.2 金标变更流程

- 金标测试失败不允许修改 `tests/golden/expected/`
- 变更走独立 PR,两人 review,说明原因
- MIGRATION.md 记录条目

## 7. MIGRATION.md 的 Known Differences

任何可接受的差异必须记录,格式:

```markdown
## Known Differences

### [2026-05-15] MACD histogram 精度差 1e-8

**影响**:`ta.macd().hist` 与 pandas-ta-classic 原版最后一位小数略有偏差

**原因**:trade-learn 在 EMA 初始化用 SMA(backtrader 默认);pandas-ta-classic 用第一个值

**影响评估**:信号时点完全一致,数值差异 < 1e-8,不影响决策

**测试**:`tests/consistency/test_indicators_core.py::test_macd_hist[...]`
rtol 从 `1e-10` 放宽到 `1e-8`
```

**每一条差异必须四字段齐全**:影响 / 原因 / 评估 / 测试。

## 8. 兼容模式(compat="1.x")

边界场景若无法完美对齐 1.x,提供兼容开关:

```python
cerebro = Cerebro(compat="1.x")     # 切回 1.x 行为
```

**谨慎使用**,每开一个 compat 分支增加维护成本。1.1 后下线。

## 9. 1.x oracle 作为 CI 第一公民

整个重构期,`reference/tradelearn_1x/` 在 CI 里跑:

```yaml
- name: 1.x oracle regression
  run: |
    cd reference/
    pip install -e tradelearn_1x
    pytest reference/tests/smoke/
```

**1.x 始终可跑**,保证对照可用。1.0 发版后下线。

## 10. 性能回归守护

不只正确性,速度也不能倒退:

```yaml
- name: Benchmark
  run: |
    uv run pytest benchmarks/ --benchmark-json=bench.json
    # 对比基线,回归超 20% 失败
    python scripts/check_benchmark_regression.py bench.json --baseline=bench-baseline.json
```

benchmark 结果归档到 GitHub Pages,公开可见。

## 11. 验证节点

| 节点 | 验证动作 |
|---|---|
| M0.5(0.1-alpha) | 金标基线固化(expected/v1.0/) |
| M1(0.1) | metrics 层全部 `rtol=1e-10` 通过 |
| M2(0.2) | factor/report/ta.* 层通过 |
| M4(0.3a Rust 核) | trades 0 差异,equity rtol=1e-6 |
| M6(1.0) | 端到端金标全通过 |

## 12. 一致性失败应急

**决策层一致性失败(trades 差异)**:
- PR block
- **不修改金标**,修改代码
- 差异根本原因定位后走正常 debug 流程

**数值层飘了 rtol**:
- 先查是否已记录在 MIGRATION.md
- 若未记录 = bug,修代码
- 若"是合理的新差异",补记录 + 放宽 rtol

**1.x oracle 跑不起来**:
- 查 reference/ 是否被误改
- 查依赖是否有不兼容升级(lock 住)

## 13. 不做的事

- ❌ 自动"发现差异自动放宽 rtol"(人工判断必需)
- ❌ 允许"reference 也要升级"(reference 整个重构期冻结)
- ❌ 差异可以"以后再查"(每条必须当时解释)
- ❌ 本地绕过 CI 强合 PR(required check 不能 override)
