# 优化指南

研究搜索复用 `tradelearn.optimize.OptunaSearch` / `OptunaBacktestSearch`，小网格使用独立 `tradelearn.integrations.search.grid_search` 或 Lite 的 `Backtest.optimize()`。它们不属于 Backtrader 的 Engine 根接口，也不增加新的搜索 DSL。可选依赖统一为 `pip install 'trade-learn[research]'`。

## Optuna：原生 trial 与 Study

```python
from tradelearn.optimize import OptunaSearch


def objective(trial):
    lookback = trial.suggest_int("lookback", 10, 60)
    threshold = trial.suggest_float("threshold", .1, 2.)
    return evaluate_validation(lookback=lookback, threshold=threshold)


study = OptunaSearch(objective, n_trials=20, direction="maximize").run(
    n_jobs=1, timeout=60,
)
print(study.best_params)
print(study.trials)  # 保留全部完成、失败、剪枝试验，不能只留最优结果
```

`evaluate_validation` 是用户显式提供的训练/验证评价函数。预处理、筛选和模型只 train-fit，Test 不参与选型；使用真实 label_end purge。失败策略、随机种子、并行/时间预算和完整 trial 信息需显式配置和保存，见[完整研究接线](research.md)。`OptunaBacktestSearch` 保留既有 `backtest_factory / param_space / metric` 接口，不自动选择金融目标或重跑未提供的评价。

## 独立 grid 与 Lite

```python
from tradelearn.integrations.search import grid_search

results = grid_search(
    MyStrategy,
    bars,  # 单标的 OHLCV DataFrame
    {"fast": [5, 10], "slow": [20, 40]},
)
for result in results:
    print(result.params, result.strategy.stats.summary)
```

可显式传 `mlflow={"experiment_name": "grid-study", ...}`；该集成在全部策略完成后一次记录已有结果，沿共同原生 Run 复用/冲突契约，不创建隐式 nested Run。普通 Analyzer 结果保留在 `result.analyzer_results`。

```python
results = backtest.optimize(fast=[5, 10], slow=[20, 40])
```

Lite 简化搜索的参数与返回仍以其当前接口为准；复杂研究使用显式 Optuna 目标和独立训练/验证链路，不把独立搜索重新放回 Engine 根 facade。
