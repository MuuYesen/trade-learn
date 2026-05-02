"""Machine-readable report artifact helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from bokeh.embed import file_html
from bokeh.resources import INLINE

from tradelearn.report.reporter import Reporter


def write_artifact_bundle(
    stats: Any,
    directory: str | Path,
    *,
    strategy: Any | None = None,
    market_data: pd.DataFrame | None = None,
    log_report: bool = True,
    log_plot: bool = False,
) -> list[Path]:
    """Write report artifacts and return the generated file paths.

    This helper owns artifact materialization for both HTML reports and
    MLflow logging. MLflow-specific code should only upload the returned files.
    """

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(stats, market_data=market_data)

    _write_tables(output_dir, stats, strategy)

    if log_report:
        reporter.report(output_dir / "report.html")
    if log_plot:
        chart = reporter.market_replay_chart()
        if chart is not None:
            (output_dir / "plot.html").write_text(
                file_html(chart, INLINE, "Tradelearn Market Replay")
            )

    _remove_report_sidecars(output_dir)
    return sorted(output_dir.iterdir())


def market_data_from_strategy(strategy: Any) -> pd.DataFrame | None:
    """Return primary market data from a strategy when available."""

    data = getattr(strategy, "data", None)
    frame = getattr(data, "_frame", None)
    if frame is None:
        return None
    return pd.DataFrame(frame).copy()


def _remove_report_sidecars(output_dir: Path) -> None:
    keep = {"artifacts.xlsx", "report.html", "plot.html"}
    keep_dirs = {"csv"}
    for path in output_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
        elif path.is_dir() and path.name not in keep_dirs:
            shutil.rmtree(path)


def _write_core_tables(output_dir: Path, stats: Any) -> None:
    _write_tables(output_dir, stats, strategy=None)


def _write_tables(output_dir: Path, stats: Any, strategy: Any | None) -> None:
    sheets = _artifact_sheets(stats, strategy)
    if not sheets:
        return
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    with pd.ExcelWriter(output_dir / "artifacts.xlsx") as writer:
        for name, frame in sheets.items():
            safe = _excel_safe_frame(frame)
            safe.to_excel(writer, sheet_name=name, index=False)
            safe.to_csv(csv_dir / f"{_csv_name(name)}.csv", index=False)


def _csv_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _artifact_sheets(stats: Any, strategy: Any | None) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    summary = _stats_field(stats, "summary", {})
    if summary:
        sheets["summary"] = pd.DataFrame(
            [{"key": key, "value": value} for key, value in dict(summary).items()]
        )
    equity = _coerce_series(_stats_field(stats, "equity", pd.Series(dtype="float64")))
    if not equity.empty:
        sheets["equity"] = _series_frame(equity, index_name="datetime", value_name="equity")
    trades = pd.DataFrame(_stats_field(stats, "trades", pd.DataFrame()))
    sheets["trades"] = trades
    positions = pd.DataFrame(_stats_field(stats, "positions", pd.DataFrame()))
    if not positions.empty:
        sheets["positions"] = positions
    fills = pd.DataFrame(_stats_field(stats, "fills", pd.DataFrame()))
    if not fills.empty:
        sheets["fills"] = fills
    weights = _research_weights(strategy)
    if weights is not None and not weights.empty:
        sheets["weights"] = _series_frame(weights, index_name="symbol", value_name="weight")
    research = _research_payload(strategy)
    if research:
        sheets["research"] = pd.DataFrame(
            [{"key": key, "value": value} for key, value in research.items()]
        )
    return sheets


def _research_weights(strategy: Any | None) -> pd.Series | None:
    result = _first_attr(
        strategy,
        ("research_result", "research_result_"),
    )
    if result is None:
        return None
    weights = getattr(result, "weights", None)
    if weights is None:
        return None
    if hasattr(weights, "raw"):
        weights = weights.raw
    return pd.Series(weights, dtype="float64", name="weight")


def _research_payload(strategy: Any | None) -> dict[str, Any]:
    result = _first_attr(strategy, ("research_result", "research_result_"))
    if result is None or not hasattr(result, "to_dict"):
        return {}
    payload = result.to_dict()
    if not isinstance(payload, dict):
        return {}
    research = {
        "name": payload.get("name"),
        "selected": ",".join(payload.get("result", {}).get("selected", [])),
        "steps": ",".join(step.get("name", "") for step in payload.get("steps", [])),
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        research.update(_flatten_mapping("artifacts", artifacts))
    return research


def _flatten_mapping(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(name, value))
        else:
            flattened[name] = _display_value(value)
    return flattened


def _display_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return value


def _coerce_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    return pd.Series(values)


def _series_frame(values: pd.Series, *, index_name: str, value_name: str) -> pd.DataFrame:
    frame = values.rename(value_name).to_frame()
    frame.index.name = frame.index.name or index_name
    return frame.reset_index()


def _excel_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = pd.DataFrame(frame).copy()
    for column in safe.columns:
        series = safe[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_convert(None)
        elif series.dtype == "object":
            safe[column] = series.map(_excel_safe_value)
    return safe


def _excel_safe_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
        return value.tz_convert(None)
    return value


def _stats_field(stats: Any, name: str, default: Any) -> Any:
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None
