# Optimization Guide

trade-learn 有两层优化入口。

## Optuna：推荐用户入口

`tradelearn.optimize` 是参数搜索的推荐入口，适合研究参数、模型参数和策略参数。

```python
from tradelearn.optimize import optimize

result = optimize(
    objective,
    search_space={
        "lookback": ("int", 10, 60),
        "threshold": ("float", 0.1, 2.0),
    },
    n_trials=100,
)
```

## Engine / Lite 的轻量 grid

Engine 的 `grid_search()` 和 Lite 的 `Backtest.optimize()` 只是 facade sugar，用来快速跑小规模网格，不建议承载复杂投研优化逻辑。

```python
results = bt.grid_search(cerebro, MyStrategy, fast=[5, 10], slow=[20, 40])
```

```python
results = backtest.optimize(fast=[5, 10], slow=[20, 40])
```
