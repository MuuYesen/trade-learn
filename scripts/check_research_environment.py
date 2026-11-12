"""Run a local, synthetic Notebook/report/MLflow acceptance without a provider."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    """Execute a real kernel using this interpreter and write reviewable artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import nbformat
    from nbclient import NotebookClient

    source = '''
from pathlib import Path
import json
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from tradelearn.research import ResearchRun
from tradelearn.lite import Backtest, Strategy
from tradelearn.report import Reporter
from tradelearn.integrations.mlflow import log

output = Path(OUTPUT)
model = LinearRegression()
X = pd.DataFrame({"x": [0., 1., 2.]})
y = pd.Series([0., 2., 4.])
with ResearchRun("offline-fixture") as run:
    run.register("model", model)
    run.call("model", "fit", X, y)
    predictions = run.call("model", "predict", X)
    research = run.finish(model=model, artifacts={"validation": "synthetic-fixture"})
assert list(predictions.round(8)) == [0., 2., 4.]
assert len(research.steps) == 2
class Hold(Strategy):
    def next(self):
        if not self.position():
            self.buy(size=1)
bars = pd.DataFrame({
    "open":[10.,11.,12.,13.,14.,15.],
    "high":[11.,12.,13.,14.,15.,16.],
    "low":[9.,10.,11.,12.,13.,14.],
    "close":[10.,11.,12.,13.,14.,15.],
    "volume":[100.]*6,
},index=pd.date_range("2026-01-01",periods=6))
backtest = Backtest(bars, Hold)
result = backtest.run()
reporter = Reporter(result)
html = reporter.report(output / "report.html")
excel = reporter.report(output / "report.xlsx")
# Export/logging must not execute an additional fit or backtest.
def forbidden(*args, **kwargs):
    raise AssertionError("report/log attempted to recompute")
model.fit = forbidden
backtest.run = forbidden
mlflow.set_tracking_uri("sqlite:///" + str(output / "tracking.db"))
mlflow.set_experiment("tl4-offline-acceptance")
with mlflow.start_run(run_name="native") as active:
    recorded = log(result, research_result=research, attachments=[html, excel])
    assert recorded.error is None, recorded
    assert recorded.run_id == active.info.run_id
    assert mlflow.active_run().info.run_id == active.info.run_id
assert mlflow.get_run(recorded.run_id).info.status == "FINISHED"
summary={"run_id":recorded.run_id,"external_run_preserved":True,
         "research_steps":len(research.steps),"html":html.name,"excel":excel.name,
         "fixture_only":True,"provider_called":False}
(output / "acceptance.json").write_text(json.dumps(summary,indent=2))
print(summary)
'''.replace("OUTPUT", repr(str(output)))
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    previous = os.environ.get("JUPYTER_PATH")
    with tempfile.TemporaryDirectory(prefix="tl4-kernel-") as directory:
        root = Path(directory)
        spec = root / "kernels" / "tl4-research"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
            "display_name":"TL4 local acceptance", "language":"python",
        }))
        os.environ["JUPYTER_PATH"] = str(root)
        try:
            client = NotebookClient(notebook, kernel_name="tl4-research", timeout=90,
                                    resources={"metadata":{"path":str(output)}})
            client.execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    nbformat.write(notebook, output / "acceptance.ipynb")
    print((output / "acceptance.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
