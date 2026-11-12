"""Cross-domain report contracts exercised through the actual convenience APIs."""
import pandas as pd


def test_data_explorer_reuses_completed_tables_between_formats(tmp_path, monkeypatch):
    from tradelearn.data import DataExplorer

    calls = []
    original = DataExplorer.describe

    def counted(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(DataExplorer, "describe", counted)
    explorer = DataExplorer(pd.DataFrame({"value": [1., 2., 3.]}))
    explorer.report(tmp_path / "exploration.html")
    explorer.report(tmp_path / "exploration.xlsx")
    assert len(calls) == 1


def test_factor_excel_only_does_not_compute_html_figures(tmp_path, monkeypatch):
    from tradelearn.factor import FactorAnalyzer

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["AAA", "BBB", "CCC"]],
        names=["timestamp", "symbol"],
    )
    factor = pd.Series([1., 2., 3.] * 4, index=index, name="factor")
    forward = pd.Series([.1, -.05, .02] * 4, index=index, name="forward")

    def forbidden(self):
        raise AssertionError("Excel-only export must not build HTML figures")

    monkeypatch.setattr(FactorAnalyzer, "plot", forbidden)
    result = FactorAnalyzer(factor, forward_returns=forward, quantiles=2).report(
        tmp_path / "factor.xlsx"
    )
    assert result.is_file()
