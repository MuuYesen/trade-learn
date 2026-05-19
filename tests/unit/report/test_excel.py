"""Tests for Excel report export."""

import xml.etree.ElementTree as ET
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


def test_reporter_report_dispatches_excel_by_suffix(tmp_path) -> None:
    """Reporter.report writes Excel when the output suffix is .xlsx."""
    path = tmp_path / "report.xlsx"

    result = Reporter(_stats(), periods=252).report(path)

    assert result == path
    assert path.exists()
    assert "summary" in _sheet_names(path)


def test_reporter_excel_appends_custom_section_sheets(tmp_path) -> None:
    """Reporter.excel appends DataFrame sheets from user-defined sections."""
    path = tmp_path / "custom-report.xlsx"

    class ExposureSection:
        name = "custom_exposure"

        def excel(self, ctx):
            assert ctx.stats is not None
            assert not ctx.returns.empty
            return {
                "custom_exposure": pd.DataFrame(
                    {"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]}
                )
            }

    Reporter(_stats(), periods=252).excel(path, sections=[ExposureSection()])

    assert "custom_exposure" in _sheet_names(path)
    assert "AAA" in _shared_strings(path)


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

    assert "active_returns" in _sheet_names(path)
    assert "active_weights" in _sheet_names(path)
    assert "performance_attr" in _sheet_names(path)
    assert "rolling_beta" in _sheet_names(path)
    workbook_text = _workbook_text(path)
    assert "alpha" in workbook_text
    assert "beta" in workbook_text
    assert "information_ratio" in workbook_text
    assert "active_return" in workbook_text
    assert "tracking_error" in workbook_text


def test_reporter_excel_drawdowns_sheet_uses_top_drawdown_table(tmp_path) -> None:
    """Reporter.excel writes the spec's top drawdowns table, not raw drawdown series."""
    path = tmp_path / "drawdowns-report.xlsx"

    Reporter(_stats(), periods=252).excel(path)

    workbook_text = _workbook_text(path)
    for header in ["peak", "valley", "recovery", "max_drawdown", "duration"]:
        assert header in workbook_text
    assert "drawdown" not in _shared_strings(path)


def test_reporter_excel_formats_return_values_to_six_decimals(tmp_path) -> None:
    """Reporter.excel applies the spec's six-decimal number format to return sheets."""
    path = tmp_path / "formatted-report.xlsx"

    Reporter(_stats(), periods=252).excel(path)

    daily_returns_style = _cell_style(path, sheet="sheet3", cell="B2")
    num_formats = _number_formats_by_style(path)
    assert num_formats[daily_returns_style] == "0.000000"


def test_reporter_excel_flattens_nested_config_parameters(tmp_path) -> None:
    """Reporter.excel expands nested config and strategy parameters into key/value rows."""
    path = tmp_path / "config-report.xlsx"
    stats = _stats()
    stats.config = {
        "strategy": {"name": "demo", "params": {"fast": 5, "slow": 20}},
        "broker": {"cash": 100_000, "commission": 0.001},
        "run_id": "run-001",
    }

    Reporter(stats, periods=252).excel(path)

    shared_strings = _shared_strings(path)
    for key in [
        "strategy.name",
        "strategy.params.fast",
        "strategy.params.slow",
        "broker.cash",
        "broker.commission",
        "run_id",
    ]:
        assert key in shared_strings
    assert "params" not in shared_strings


def test_reporter_excel_preserves_numeric_config_values(tmp_path) -> None:
    """Reporter.excel keeps numeric config values numeric after flattening."""
    path = tmp_path / "numeric-config-report.xlsx"
    stats = _stats()
    stats.config = {
        "strategy": {"params": {"fast": 5}},
        "broker": {"cash": 100_000, "commission": 0.001},
    }

    Reporter(stats, periods=252).excel(path)

    assert _cell_type(path, sheet="config", cell="B2") != "s"
    assert _cell_type(path, sheet="config", cell="B3") != "s"
    assert _cell_type(path, sheet="config", cell="B4") != "s"


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


def _cell_style(path, sheet: str, cell: str) -> int:
    with ZipFile(path) as workbook:
        xml = workbook.read(_worksheet_path(path, sheet))
    root = ET.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    element = root.find(f".//x:c[@r='{cell}']", namespace)
    assert element is not None
    return int(element.attrib["s"])


def _cell_type(path, sheet: str, cell: str) -> str:
    with ZipFile(path) as workbook:
        xml = workbook.read(_worksheet_path(path, sheet))
    root = ET.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    element = root.find(f".//x:c[@r='{cell}']", namespace)
    assert element is not None
    return element.attrib.get("t", "")


def _worksheet_path(path, sheet: str) -> str:
    if sheet.startswith("sheet"):
        return f"xl/worksheets/{sheet}.xml"
    names = _sheet_names(path)
    return f"xl/worksheets/sheet{names.index(sheet) + 1}.xml"


def _number_formats_by_style(path) -> dict[int, str]:
    with ZipFile(path) as workbook:
        xml = workbook.read("xl/styles.xml")
    root = ET.fromstring(xml)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    custom_formats = {
        element.attrib["numFmtId"]: element.attrib["formatCode"]
        for element in root.findall(".//x:numFmt", namespace)
    }
    styles = {}
    for index, element in enumerate(root.findall(".//x:cellXfs/x:xf", namespace)):
        styles[index] = custom_formats.get(element.attrib.get("numFmtId", ""), "")
    return styles


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

    def factor_information_coefficient(self) -> pd.Series:
        """Return factor rank IC series for Excel tests."""
        return pd.Series(
            [0.15, 0.25],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="factor_information_coefficient",
        )

    def quantile_turnover(self) -> pd.DataFrame:
        """Return factor turnover series for Excel tests."""
        return pd.DataFrame(
            {1: [0.20, 0.30], 2: [0.40, 0.50]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def factor_rank_autocorrelation(self) -> pd.Series:
        """Return factor autocorrelation series for Excel tests."""
        return pd.Series(
            [0.60, 0.70],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="factor_rank_autocorrelation",
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
