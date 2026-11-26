"""Analysis reports must look like the rest of TradeLearn, not raw pandas HTML.

explore.report() / FactorAnalyzer.report() route through Reporter.from_analysis,
which rendered bare <table border=1> with 17 significant digits. Excel stays the
lossless channel; the HTML is for reading, and the exact value is still carried
on each cell so nothing is lost.
"""

from pathlib import Path

import pandas as pd

from tradelearn.report import Reporter

SOURCE = {
    "title": "TradeLearn Data Profile",
    "tables": {
        "numeric": pd.DataFrame(
            {"column": ["close"], "mean": [0.123456789012345], "count": [4353]}
        )
    },
    "metrics": {"coverage": {"value": 0.9876543210987654, "unit": "ratio", "reason": ""}},
    "metadata": {"rows": 4353, "columns": 11},
}


def _html(tmp_path: Path, source=None) -> str:
    reporter = Reporter.from_analysis(source or SOURCE)
    return Path(reporter.report(tmp_path / "analysis.html")).read_text(encoding="utf-8")


def test_analysis_html_carries_the_shared_report_styling(tmp_path):
    content = _html(tmp_path)
    assert "<style>" in content, "analysis report must ship the shared stylesheet"
    assert 'class="shell"' in content and 'class="report-header"' in content
    assert "border=\"1\"" not in content, "raw pandas table borders must be gone"


def test_numbers_are_readable_but_the_exact_value_is_retained(tmp_path):
    content = _html(tmp_path)
    assert ">0.123457<" in content, "displayed value should be rounded for reading"
    assert ">0.123456789012345<" not in content, "full precision must not be the display text"
    assert "0.123456789012345" in content, "exact value must still be carried (cell title)"


def test_metrics_render_as_cards_with_unit_and_reason(tmp_path):
    content = _html(tmp_path)
    assert 'class="kpi-grid"' in content or 'class="kpi-card"' in content
    assert "coverage" in content and "ratio" in content


def test_metadata_and_title_are_escaped_and_shown(tmp_path):
    content = _html(tmp_path, {**SOURCE, "title": "Factor <test>"})
    assert "Factor &lt;test&gt;" in content
    assert "4353" in content and "11" in content


def test_excel_remains_the_lossless_channel(tmp_path):
    reporter = Reporter.from_analysis(SOURCE)
    sheets = pd.read_excel(reporter.report(tmp_path / "analysis.xlsx"), sheet_name=None)
    assert sheets["numeric"].iloc[0]["mean"] == 0.123456789012345


def test_existing_analysis_contract_still_holds(tmp_path):
    reporter = Reporter.from_analysis(
        {
            "tables": {"facts": pd.DataFrame()},
            "metrics": {"score": {"value": float("nan"), "unit": "ratio"}},
            "figures": {"empty": None},
        }
    )
    content = Path(reporter.html(tmp_path / "empty.html")).read_text(encoding="utf-8")
    assert "not_calculable" in content and "Not provided" in content
