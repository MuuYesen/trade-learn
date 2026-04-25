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
    assert "factor_quantiles" in _sheet_names(path)


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
