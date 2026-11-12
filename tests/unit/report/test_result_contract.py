"""Report exports retain facts and require no rendering stack for numbers."""

import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest


def test_numeric_reporter_import_does_not_require_rendering_dependencies():
    code = """
import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'bokeh','jinja2','xlsxwriter','pygwalker','mlflow'}:
   raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from tradelearn.report import Reporter
import pandas as pd
r=Reporter({'returns':pd.Series([.01,-.02]), 'trades':pd.DataFrame({'pnl':[1.,-2.]}), 'summary':{}})
assert r.summary()['total_trades']==2
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_artifact_writer_never_deletes_unrelated_files(tmp_path):
    from tradelearn.report.artifacts import write_artifact_bundle

    keep = tmp_path / "user-notes.txt"
    keep.write_text("keep")
    folder = tmp_path / "user-folder"
    folder.mkdir()
    (folder / "a").write_text("keep")
    stats = SimpleNamespace(summary={"return_pct": 1.234567890123}, trades=pd.DataFrame())
    paths = write_artifact_bundle(stats, tmp_path, log_report=False)
    assert keep.read_text() == "keep" and (folder / "a").read_text() == "keep"
    assert keep not in paths and folder not in paths


def test_reporter_caches_summary_and_returns_independent_copy(monkeypatch):
    import tradelearn.metrics as metrics
    from tradelearn.report import Reporter

    calls = []
    original = metrics.annual_return
    monkeypatch.setattr(
        metrics,
        "annual_return",
        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs),
    )
    reporter = Reporter(
        {"returns": pd.Series([0.01, -0.02]), "trades": pd.DataFrame({"pnl": [1.0, -2.0]})}
    )
    first = reporter.summary()
    assert reporter.summary() == first
    assert len(calls) == 1
    first["annual_return"] = 999
    assert reporter.summary()["annual_return"] != 999


def test_empty_report_summary_preserves_missing():
    from tradelearn.report import Reporter

    summary = Reporter(
        {"returns": pd.Series(dtype=float), "trades": pd.DataFrame({"pnl": pd.Series(dtype=float)})}
    ).summary()
    assert pd.isna(summary["cumulative_return"])
    assert pd.isna(summary["sharpe_ratio"])


def test_report_metric_facts_preserve_units_precision_and_missing():
    from tradelearn.report import Reporter

    reporter = Reporter(
        {
            "returns": pd.Series([0.01234567890123, -0.02]),
            "trades": pd.DataFrame({"pnl": [1.0, -2.0]}),
        }
    )
    facts = reporter.metric_facts().set_index("name")
    assert facts.loc["annual_return", "unit"] == "ratio"
    assert facts.loc["annual_return", "value"] == reporter.summary()["annual_return"]
    assert facts.loc["turnover", "reason"] == "not_provided"
    assert pd.isna(facts.loc["turnover", "value"])


def test_duplicate_custom_excel_names_rejected_before_overwrite(tmp_path):
    from tradelearn.report import Reporter

    reporter = Reporter(
        {
            "returns": pd.Series([0.01, -0.02], index=pd.date_range("2026-01-01", periods=2)),
            "trades": pd.DataFrame({"pnl": [1.0, -2.0]}),
        }
    )
    with pytest.raises(ValueError, match="sheet"):
        reporter.excel(
            tmp_path / "collision.xlsx",
            sections=[{"excel": {"summary": pd.DataFrame({"overwrite": [1]})}}],
        )


def test_report_version_follows_loaded_module_not_historical_distribution(monkeypatch):
    import tradelearn
    import tradelearn.report.html as html

    monkeypatch.setattr(html,"version",lambda name:"0.1.2.0",raising=False)
    assert html._package_version() == tradelearn.__version__


def test_trade_pnl_basis_is_explicit_and_original_frame_is_preserved():
    from tradelearn.report import Reporter

    trades = pd.DataFrame({"pnl": [100., -50.], "pnlcomm": [98., -52.]})
    source = {"returns": pd.Series([0., .01]), "trades": trades}
    gross = Reporter(source)
    assert gross.summary()["profit_factor"] == 2.
    source["metric_metadata"] = {"profit_factor": {"pnl_basis": "pnlcomm"}}
    net = Reporter(source)
    assert net.summary()["profit_factor"] == 98. / 52.
    assert net.metric_facts().set_index("name").loc["profit_factor", "pnl_basis"] == "pnlcomm"
    assert trades["pnl"].tolist() == [100., -50.]
