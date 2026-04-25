"""Tests for Excel report export."""

from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd

from tradelearn import metrics
from tradelearn.report import Reporter


def test_reporter_excel_writes_spec_sheets(tmp_path) -> None:
    """Reporter.excel writes a multi-sheet xlsx report using xlsxwriter."""
    path = tmp_path / "report.xlsx"
    stats = _stats()

    result = Reporter(stats, periods=252).excel(path)

    assert result == path
    assert path.exists()
    assert _sheet_names(path) == [
        "summary",
        "trades",
        "daily_returns",
        "monthly_returns",
        "drawdowns",
        "positions",
        "orders",
        "config",
    ]


def test_reporter_excel_accepts_mapping_stats(tmp_path) -> None:
    """Reporter.excel accepts dict-shaped stats."""
    path = tmp_path / "mapping-report.xlsx"
    stats = {
        "returns": _returns(),
        "trades": _trades(),
        "positions": pd.DataFrame(),
        "orders": pd.DataFrame(),
        "summary": {"strategy_name": "demo"},
        "config": {"strategy": "demo"},
    }

    Reporter(stats).excel(path)

    assert path.exists()


def test_reporter_excel_writes_factor_quantile_sheet_when_analyzer_exists(tmp_path) -> None:
    """Reporter.excel writes factor quantile returns when a factor analyzer exists."""
    path = tmp_path / "factor-report.xlsx"
    stats = {
        "returns": _returns(),
        "trades": _trades(),
        "analyzers": {"factor": _FactorAnalyzerStub()},
    }

    Reporter(stats).excel(path)

    assert "factor_ic" in _sheet_names(path)
    assert "factor_rank_ic" in _sheet_names(path)
    assert "factor_turnover" in _sheet_names(path)
    assert "factor_autocorr" in _sheet_names(path)
    assert "factor_long_short" in _sheet_names(path)
    assert "factor_quantiles" in _sheet_names(path)


def test_reporter_excel_accepts_benchmark_series(tmp_path) -> None:
    """Reporter.excel writes benchmark metrics and rolling beta when provided."""
    path = tmp_path / "benchmark-report.xlsx"

    Reporter(_stats(), periods=252).excel(path, benchmark=_benchmark())

    assert "rolling_beta" in _sheet_names(path)
    workbook_text = _workbook_text(path)
    assert "alpha" in workbook_text
    assert "beta" in workbook_text
    assert "information_ratio" in workbook_text


def test_reporter_excel_drawdowns_sheet_uses_top_drawdown_table(tmp_path) -> None:
    """Reporter.excel writes the spec's top drawdowns table, not raw drawdown series."""
    path = tmp_path / "drawdowns-report.xlsx"

    Reporter(_stats(), periods=252).excel(path)

    workbook_text = _workbook_text(path)
    for header in ["peak", "valley", "recovery", "max_drawdown", "duration"]:
        assert header in workbook_text
    assert "drawdown" not in _shared_strings(path)


def _sheet_names(path) -> list[str]:
    with ZipFile(path) as workbook:
        xml = workbook.read("xl/workbook.xml").decode()
    names = []
    for chunk in xml.split("<sheet ")[1:]:
        marker = 'name="'
        start = chunk.index(marker) + len(marker)
        end = chunk.index('"', start)
        names.append(chunk[start:end])
    return names


def _workbook_text(path) -> str:
    with ZipFile(path) as workbook:
        return "\n".join(
            workbook.read(name).decode(errors="ignore")
            for name in workbook.namelist()
            if name.endswith(".xml")
        )


def _shared_strings(path) -> set[str]:
    with ZipFile(path) as workbook:
        xml = workbook.read("xl/sharedStrings.xml").decode(errors="ignore")
    values = set()
    for chunk in xml.split("<t>")[1:]:
        values.add(chunk.split("</t>", 1)[0])
    return values


def _stats() -> SimpleNamespace:
    returns = _returns()
    return SimpleNamespace(
        returns=returns,
        equity=metrics.cum_returns(returns, starting_value=100_000.0),
        trades=_trades(),
        positions=pd.DataFrame({"symbol": ["AAA", "BBB"], "value": [10_000.0, 12_000.0]}),
        orders=pd.DataFrame({"symbol": ["AAA"], "size": [100]}),
        summary={},
        analyzers={},
        config={"strategy": "demo"},
    )


def _returns() -> pd.Series:
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="returns",
    )


def _benchmark() -> pd.Series:
    return pd.Series(
        [0.01, -0.005, 0.01, -0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="benchmark",
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame({"pnl": [100.0, -50.0, 25.0, -10.0]})


class _FactorAnalyzerStub:
    def ic(self) -> pd.Series:
        """Return factor IC series for Excel tests."""
        return pd.Series(
            [0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="ic",
        )

    def rank_ic(self) -> pd.Series:
        """Return factor rank IC series for Excel tests."""
        return pd.Series(
            [0.15, 0.25],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="rank_ic",
        )

    def turnover(self) -> pd.Series:
        """Return factor turnover series for Excel tests."""
        return pd.Series(
            [0.30, 0.40],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="turnover",
        )

    def autocorrelation(self) -> pd.Series:
        """Return factor autocorrelation series for Excel tests."""
        return pd.Series(
            [0.60, 0.70],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="autocorrelation",
        )

    def quantile_cumulative_returns(self) -> pd.DataFrame:
        """Return factor quantile cumulative returns for Excel tests."""
        return pd.DataFrame(
            {1: [0.01, 0.02], 2: [0.03, 0.04]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def long_short_cumulative_returns(self) -> pd.DataFrame:
        """Return factor long-short cumulative returns for Excel tests."""
        return pd.DataFrame(
            {
                "long": [0.03, 0.04],
                "short": [0.01, 0.02],
                "spread": [0.02, 0.02],
            },
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )
