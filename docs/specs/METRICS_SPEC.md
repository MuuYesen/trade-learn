# METRICS SPEC

`tradelearn.metrics` 模块规格——**全项目指标唯一真源**,融合 empyrical / pyfolio / alphalens 共 35 个核心指标。

## 1. 设计原则

1. **纯函数**:无副作用,不依赖全局状态
2. **显式口径**:年化因子、收益定义、NaN 策略必须参数化或固定
3. **Numpy-style docstring**:带 Examples(doctest 可跑)
4. **金标对照**:与 empyrical/pyfolio/alphalens 原版 `rtol=1e-10`

## 2. 口径统一约定

### 2.1 年化因子(periods)

**必填参数**,无默认值。调用方显式声明:

```python
sharpe(returns, periods=252)   # 日频股票
sharpe(returns, periods=52)    # 周频
sharpe(returns, periods=12)    # 月频
```

理由:empyrical 默认 252 / pyfolio 自动推断 / alphalens 用 `periods_per_year` —— 三方混淆导致 sharpe 对不上。统一强制显式。

### 2.2 收益定义

**全项目简单收益** `r_t = p_t / p_{t-1} - 1`。

对数收益只在**输入转换时**使用:

```python
from tradelearn.metrics.returns import log_to_simple
r_simple = log_to_simple(r_log)
```

### 2.3 NaN 策略

默认 `dropna`,可参数化:

```python
def sharpe(returns, periods, nan_policy="drop") -> float:
    # nan_policy: "drop" | "zero" | "propagate" | "raise"
```

### 2.4 基准对齐

涉及基准的函数(alpha/beta/information_ratio)用 `pandas.align(join='inner')`:

```python
def alpha(returns, benchmark, periods, rf=0.0) -> float:
    r, b = returns.align(benchmark, join='inner')
    ...
```

**双方缺失日都剔除**,不 forward-fill。

### 2.5 无风险利率

默认 `rf=0.0`(年化),显式传入覆盖:

```python
sharpe(returns, periods=252, rf=0.03)
```

### 2.6 浮点容忍

所有指标 bit-level reproducible:

- 不使用并行 reduce
- 固定累积顺序
- 相同输入 → 相同输出(1e-16 精度)

## 3. 模块布局

```
tradelearn/metrics/
├── __init__.py          # 对外暴露所有函数
├── returns.py           # 5 个:收益类
├── risk.py              # 13 个:风险类
├── factor.py            # 7 个:因子类
├── trade.py             # 7 个:交易类
└── _common.py           # 内部共用(年化辅助 / NaN 处理)
```

## 4. 完整指标清单(35 个)

### 4.1 returns.py(5 个)

| 函数 | 签名 | 公式 | 金标对照 |
|---|---|---|---|
| `annual_return` | `(returns, periods) -> float` | `(1+total_return)^(periods/n) - 1` | empyrical.annual_return |
| `cum_returns` | `(returns, starting_value=0) -> Series` | `cumprod(1+r) - 1` | empyrical.cum_returns |
| `simple_returns` | `(prices) -> Series` | `p / p.shift(1) - 1` | empyrical.simple_returns |
| `log_to_simple` | `(log_returns) -> Series` | `exp(r) - 1` | — |
| `excess_returns` | `(returns, rf, periods) -> Series` | `r - rf/periods` | empyrical.excess_returns |

### 4.2 risk.py(13 个)

| 函数 | 签名 | 公式 |
|---|---|---|
| `sharpe` | `(returns, periods, rf=0) -> float` | `(mean - rf/n) / std * sqrt(periods)` |
| `sortino` | `(returns, periods, rf=0) -> float` | 下行波动分母 |
| `calmar` | `(returns, periods) -> float` | `annual_return / |max_drawdown|` |
| `max_drawdown` | `(returns) -> float` | 累计收益曲线最大回撤 |
| `drawdown_series` | `(returns) -> Series` | 每日回撤值 |
| `volatility` | `(returns, periods) -> float` | `std * sqrt(periods)` |
| `downside_risk` | `(returns, periods, required=0) -> float` | 下行波动 |
| `var` | `(returns, cutoff=0.05) -> float` | 历史 VaR |
| `cvar` | `(returns, cutoff=0.05) -> float` | 条件 VaR |
| `beta` | `(returns, benchmark) -> float` | OLS 斜率 |
| `alpha` | `(returns, benchmark, periods, rf=0) -> float` | 截距,年化 |
| `information_ratio` | `(returns, benchmark, periods) -> float` | 主动收益/跟踪误差 |
| `tail_ratio` | `(returns, cutoff=0.05) -> float` | 右尾/左尾 |
| `omega` | `(returns, threshold=0, periods=252) -> float` | 上/下 gain ratio |

注:13 个,上表列 14。合并 `downside_risk` 与 `sortino` 内部调用。

### 4.3 factor.py(7 个)

用于 alphalens 融合的 IC 类指标:

| 函数 | 签名 | 说明 |
|---|---|---|
| `ic` | `(factor, forward_returns) -> Series` | 按 date 的 Pearson IC |
| `rank_ic` | `(factor, forward_returns) -> Series` | Spearman rank IC |
| `ic_ir` | `(ic_series, periods) -> float` | IC 信息比率 |
| `factor_returns` | `(factor, prices, quantiles=5) -> DataFrame` | 分组收益 |
| `quantile_returns` | `(factor, forward_returns, quantiles=5) -> DataFrame` | 分组平均 |
| `turnover` | `(factor) -> Series` | 因子换手率 |
| `autocorrelation` | `(factor, lag=1) -> Series` | 自相关 |

### 4.4 trade.py(7 个)

用于策略回测结果分析:

| 函数 | 签名 | 说明 |
|---|---|---|
| `win_rate` | `(trades) -> float` | 胜率 |
| `profit_factor` | `(trades) -> float` | `sum(wins) / sum(|losses|)` |
| `avg_win` | `(trades) -> float` | 平均盈利 |
| `avg_loss` | `(trades) -> float` | 平均亏损 |
| `max_consecutive_wins` | `(trades) -> int` | 最大连胜 |
| `max_consecutive_losses` | `(trades) -> int` | 最大连败 |
| `expectancy` | `(trades) -> float` | `win_rate*avg_win - loss_rate*|avg_loss|` |

## 5. 函数签名模板

```python
def sharpe(
    returns: pd.Series,
    periods: int,
    rf: float = 0.0,
    nan_policy: Literal["drop", "zero", "propagate", "raise"] = "drop",
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Simple returns (r_t = p_t/p_{t-1} - 1).
    periods : int
        Periods per year. 252 for daily, 52 for weekly, 12 for monthly.
    rf : float, default 0.0
        Annualized risk-free rate.
    nan_policy : str, default "drop"

    Returns
    -------
    float
        Annualized Sharpe.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> r = pd.Series(np.random.normal(0.0005, 0.01, 252))
    >>> round(sharpe(r, periods=252, rf=0.03), 4)
    0.2573

    See Also
    --------
    sortino : Downside-risk-based Sharpe variant.
    """
    ...
```

**每个函数必须有**:Parameters / Returns / Examples(doctest 可跑)。

## 6. 金标测试

### 6.1 对照源

- empyrical / pyfolio / alphalens 原版(dev 依赖,CI 临时装)

### 6.2 测试模板

```python
# tests/consistency/test_metrics_sharpe.py
import empyrical as ep
from tradelearn.metrics import sharpe

@pytest.mark.parametrize("fixture", GOLDEN_RETURNS)
def test_sharpe_matches_empyrical(fixture):
    ours = sharpe(fixture.returns, periods=252)
    theirs = ep.sharpe_ratio(fixture.returns)
    assert np.isclose(ours, theirs, rtol=1e-10)
```

### 6.3 金标数据

`tests/golden/returns/` 存 10 条典型 returns 序列(parquet),作为所有 metrics 测试的 fixture。

## 7. 性能要求

- 单条 Series(10 年日线,~2500 点)所有指标累计 < 50 ms
- 无需 numba / cython,纯 numpy 实现

## 8. 错误处理

```python
class MetricsError(TradelearnError):
    """Raised by metrics module."""

def sharpe(returns, periods, ...):
    if len(returns) < 2:
        raise MetricsError("need at least 2 periods for sharpe")
    if periods <= 0:
        raise MetricsError(f"invalid periods: {periods}")
    if returns.isna().all():
        raise MetricsError("all values are NaN")
    ...
```

错误消息必须包含:**是什么错 + 在哪里 + 怎么修**。

## 9. 独有指标(empyrical 没有)

基于 alphalens 的因子类指标,我们保留原版逻辑但调 `metrics` 底层:

```python
# tradelearn/factor/analyzer.py
from tradelearn.metrics import ic, rank_ic, ic_ir

class FactorAnalyzer:
    def ic(self) -> pd.Series:
        return ic(self.factor, self.forward_returns)   # ← 调 metrics
```

## 10. 不做的事

- ❌ 非年化版本(不要 `sharpe_daily` / `sharpe_weekly`)——靠 `periods` 参数
- ❌ 自动频率推断——用户显式传 `periods`
- ❌ 多频率自动转换——`resample` 交给用户
- ❌ 画图函数——归 `report/`
- ❌ 依赖 scipy.stats 之外的重库
