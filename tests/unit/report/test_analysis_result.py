"""Completed domain tables use one Reporter entry for HTML and Excel."""

from pathlib import Path

import pandas as pd
import pytest

from tradelearn.report import Reporter


def test_analysis_exports_snapshot_values_units_and_missing_reasons(tmp_path):
    source = {
        "title": "Factor <test>",
        "tables": {"IC": pd.DataFrame({"IC": [0.123456789012345]})},
        "metrics": {
            "coverage": {"value": None, "unit": "ratio", "reason": "no_labels"},
            "ic": {"value": 0.123456789012345, "unit": "ratio"},
        },
        "metadata": {"schema_version": "1", "periods": 365},
    }
    reporter = Reporter.from_analysis(source)
    source["tables"]["IC"].iloc[0, 0] = 999
    html = reporter.report(tmp_path / "factor.html")
    excel = reporter.report(tmp_path / "factor.xlsx")
    content = Path(html).read_text()
    assert "no_labels" in content and "ratio" in content and "365" in content
    assert "Factor &lt;test&gt;" in content
    sheets = pd.read_excel(excel, sheet_name=None)
    assert sheets["IC"].iloc[0]["IC"] == pytest.approx(0.123456789012345, abs=1e-15)
    metrics = sheets["metrics"].set_index("name")
    assert metrics.loc["coverage", "reason"] == "no_labels"
    assert pd.isna(metrics.loc["coverage", "value"])
    assert metrics.loc["ic", "unit"] == "ratio"


def test_analysis_missing_and_conflicting_sheet_names_rejected():
    with pytest.raises(ValueError, match="tables"):
        Reporter.from_analysis({"tables": {}})
    with pytest.raises(ValueError, match="sheet"):
        Reporter.from_analysis({"tables": {"a/b": pd.DataFrame(), "a:b": pd.DataFrame()}})
    with pytest.raises(ValueError, match="sheet"):
        Reporter.from_analysis({"tables": {"metrics": pd.DataFrame()}})


def test_analysis_uses_custom_sections_without_domain_recompute(tmp_path):
    reporter = Reporter.from_analysis({"tables": {"facts": pd.DataFrame({"value": [1]})}})
    section = {"html": "<p>extra content</p>", "excel": {"extra": pd.DataFrame({"extra": [2]})}}
    assert (
        "extra content" in reporter.report(tmp_path / "report.html", sections=[section]).read_text()
    )
    assert (
        "extra"
        in pd.ExcelFile(reporter.report(tmp_path / "report.xlsx", sections=[section])).sheet_names
    )


def test_analysis_excel_preserves_domain_index_identity(tmp_path):
    values = pd.Series([0.1, 0.2], index=pd.Index(["AAA", "BBB"], name="symbol"), name="IC")
    reporter = Reporter.from_analysis({"tables": {"IC": values}})
    table = pd.read_excel(reporter.excel(tmp_path / "indexed.xlsx"), sheet_name="IC")
    assert table["symbol"].tolist() == ["AAA", "BBB"]


def test_analysis_missing_figure_and_nan_metric_are_explicit(tmp_path):
    reporter = Reporter.from_analysis(
        {
            "tables": {"facts": pd.DataFrame()},
            "metrics": {"score": {"value": float("nan"), "unit": "ratio"}},
            "figures": {"empty": None},
        }
    )
    content = reporter.html(tmp_path / "empty.html").read_text()
    assert "not_calculable" in content and "Not provided" in content
    with pytest.raises(ValueError, match="scalar"):
        Reporter.from_analysis({"tables": {"facts": pd.DataFrame()}, "metrics": {"bad": [1, 2]}})
