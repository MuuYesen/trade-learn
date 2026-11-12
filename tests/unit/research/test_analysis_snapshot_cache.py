import pandas as pd

from tradelearn.data import DataExplorer
from tradelearn.factor import FactorAnalyzer


def test_explorer_cache_invalidates_on_input_and_returns_independent_tables(monkeypatch):
    calls = []
    original = DataExplorer.describe
    monkeypatch.setattr(DataExplorer, "describe", lambda self: (calls.append(1), original(self))[1])
    explorer = DataExplorer(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    first = explorer.to_analysis()
    first["tables"]["describe"].iloc[0, -1] = -999.0
    second = explorer.to_analysis()
    assert len(calls) == 1
    assert second["tables"]["describe"].iloc[0, -1] != -999.0
    explorer.data.loc[0, "x"] = 5.0
    explorer.to_analysis()
    assert len(calls) == 2


def test_factor_completed_snapshot_has_facts_and_invalidates_on_input(monkeypatch):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["A", "B", "C"]], names=["timestamp", "symbol"]
    )
    factor = pd.Series([1.0, 2.0, 3.0] * 4, index=index)
    forward = pd.Series([0.1, -0.05, 0.02] * 4, index=index)
    analyzer = FactorAnalyzer(factor, forward_returns=forward, quantiles=2)
    calls = []
    original = FactorAnalyzer.summary
    monkeypatch.setattr(
        FactorAnalyzer, "summary", lambda self: (calls.append(1), original(self))[1]
    )
    first = analyzer.to_analysis()
    analyzer.to_analysis()
    assert len(calls) == 1
    assert first["metrics"]["ic_mean"]["unit"] == "correlation"
    assert first["metadata"]["sample_coverage"]["paired_observations"] == 12
    factor.iloc[0] = 2.5
    analyzer.to_analysis()
    assert len(calls) == 2


def test_factor_html_then_excel_reuses_diagnostics_and_figure(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["A", "B", "C"]], names=["timestamp", "symbol"]
    )
    analyzer = FactorAnalyzer(
        pd.Series([1.0, 2.0, 3.0] * 4, index=index),
        forward_returns=pd.Series([0.1, -0.05, 0.02] * 4, index=index),
        quantiles=2,
    )
    calls = {"summary": 0, "plot": 0}
    original_summary, original_plot = FactorAnalyzer.summary, FactorAnalyzer.plot

    def summary(self):
        calls["summary"] += 1
        return original_summary(self)

    def plot(self):
        calls["plot"] += 1
        return original_plot(self)

    monkeypatch.setattr(FactorAnalyzer, "summary", summary)
    monkeypatch.setattr(FactorAnalyzer, "plot", plot)
    analyzer.report(tmp_path / "factor.html")
    analyzer.report(tmp_path / "factor.xlsx")
    analyzer.report(tmp_path / "factor-again.html")
    assert calls == {"summary": 1, "plot": 1}
