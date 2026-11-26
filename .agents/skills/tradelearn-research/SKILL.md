---
name: tradelearn-research
description: Use for local TradeLearn research notebooks, reports and experiment logging.
---

Target package: trade-learn 0.3.6. Check the installed version before using this reference.
Keep Bars/features as Pandas objects. Split time and purge using actual label_end before fitting.
Define components before execution; explicitly register them and record actual calls in ResearchRun:

```python
from tradelearn.research import ResearchRun
with ResearchRun("validation") as run:
    run.register("model", model)
    run.call("model", "fit", X_train, y_train)
```

Use `tradelearn.lite.Backtest` for convenience or Backtrader-shaped `tradelearn.engine` for control.
A report consumes completed facts. For a completed strategy result:

```python
from tradelearn.report import Reporter
Reporter(result).report("report.html")
from tradelearn.integrations.mlflow import log
logged = log(result, research_result=research_result)
```

Use Reporter.from_analysis for completed domain tables/metrics/metadata.
Do not fit or refetch during rendering.
MLflow records only supplied results. Reuse compatible native active runs.
Do not start nested runs automatically.
Model publication and attachments are explicit.
Preserve fitted Pipeline/FeatureSet/schema/versions for restoration.
Work on local research inputs under the user's task authorization.
Do not start live trading or paid provider calls from this skill.
