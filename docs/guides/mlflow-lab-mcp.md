# MLflow / JupyterLab / MCP

## MLflow

Lite:

```python
stats = bt.run()
bt.log_mlflow(
    experiment_name="tradelearn-research",
    run_name="lite-index-enhance",
    artifacts=["report.html", "plot.html"],
)
```

Engine:

```python
cerebro.addanalyzer(
    bt.analyzers.MLflowAnalyzer,
    _name="mlflow",
    experiment_name="tradelearn-research",
)
```

MLflow 适合记录：

- 参数和核心指标。
- report.html / plot.html。
- CSV artifacts。
- XLSX artifacts。
- research steps 和 allocator 参数。

## JupyterLab

`tradelearn.lab` 用于本地研究环境启动计划，适合把 notebook、MLflow、MCP 工具放到同一研究工作台里。

## MCP

`tradelearn.mcp` 面向自动化工具和智能助手集成，用于检索项目、配置、API、lab plan 等能力。它不承载策略语义，也不反向进入回测 runtime。
