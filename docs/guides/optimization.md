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

## 共同试验执行与记录（0.3.5）

四个入口保留各自外观：Engine 使用 `cerebro.optstrategy(MyStrategy, period=[10, 20])` 后 `cerebro.run()`，默认返回 `list[list[OptReturn]]`，`optreturn=False` 返回 `list[list[Strategy]]`；Lite `Backtest.optimize()` 仍按 `return_pct` 返回最优 `LiteStats`；独立 `grid_search()` 返回 `list[GridSearchResult]`；Optuna 两种便利对象的 `run()` 仍返回原生 `Study`。参数枚举、空网格和原异常不改成统一 DSL。Optuna 只在实际调用时导入，普通网格无需安装它。

每轮复用原单次历史回测，在试验边界物化完整策略默认参数，再隔离 feed、broker、费用／滑点模型和策略运行状态，保实际现金、费率、自定义费用／滑点、COC、匹配模式、Sizer、Analyzer、Observer 和已有数据配置。搜索不连接 Store／实时 feed，不执行实时优化；不启动另一撮合循环或进程池。隔离不沙箱化用户代码：用户函数自己写外部文件、修改类级全局变量等副作用仍由用户负责。

Engine／Lite 搜索对象的 `trial_records` 保存全部已知候选的记录，独立网格每个结果的 `trial_record` 对应同一结构。记录包括选参、含默认值的策略配置、数据值／索引指纹、列／dtype／来源属性、有效执行配置、评价及结果摘要。失败仍抛出原异常；Engine／Lite 对象保记录，网格的原异常附 `trial_records`，尚未执行的有限网格候选记 `cancelled/search_aborted`，不伪称已运行。`KeyboardInterrupt` 等取消保持原异常。Optuna 保原生失败／剪枝策略；`trial_records` 与 `Study.user_attrs['tradelearn.trials']` 对齐其终态，SQLite 恢复保留旧记录；没有生成的未来 Optuna 参数不编造为试验。

独立网格需要非默认回测配置时，显式提供现有 Cerebro 的工厂；每轮依然使用快照副本，不消耗工厂原对象：

```python
from tradelearn import engine as bt
from tradelearn.integrations.search import grid_search
from tradelearn.research import ResearchRun


def configured_cerebro():
    cerebro = bt.Cerebro(stdstats=False, runonce=False)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.001)
    return cerebro


with ResearchRun("historical-grid") as research:
    results = grid_search(
        MyStrategy, bars, {"period": [10, 20]},
        cerebro_factory=configured_cerebro,
    )
    result = research.finish()

# 按需、显式记录已有结果；不重跑、不训练、不自动连接远端。
from tradelearn.integrations.mlflow import log
tracking = log(result, uri="sqlite:///research.db")
```

`ResearchRun` 的 `search.trial` 步骤保全部记录。显式 MLflow `log()` 将完整脱敏记录写为 `search/trials.json`（凭据字段剔除），沿既有 Run 复用／冲突规则；独立网格的 `mlflow={...}` 同样记录完成、失败及取消候选。追踪写入状态与试验本身状态分别保留；追踪错误不覆盖已发生的原试验异常。默认不创建 MLflow Run、不联网、不上传原行情。
