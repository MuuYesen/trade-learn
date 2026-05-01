"""Machine-readable report artifact helpers."""

from __future__ import annotations

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
    log_report: bool = False,
    log_plot: bool = False,
) -> list[Path]:
    """Write report artifacts and return the generated file paths.

    This helper owns artifact materialization for both HTML reports and
    MLflow logging. MLflow-specific code should only upload the returned files.
    """

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(stats, market_data=market_data)

    _write_core_tables(output_dir, stats)
    _write_pipeline_weights(output_dir, strategy)

    if log_report:
        reporter.report(output_dir / "report.html")
    if log_plot:
        chart = reporter.market_replay_chart()
        if chart is not None:
            (output_dir / "plot.html").write_text(
                file_html(chart, INLINE, "Tradelearn Market Replay")
            )

    return sorted(path for path in output_dir.iterdir() if path.is_file())


def market_data_from_strategy(strategy: Any) -> pd.DataFrame | None:
    """Return primary market data from a strategy when available."""

    data = getattr(strategy, "data", None)
    frame = getattr(data, "_frame", None)
    if frame is None:
        return None
    return pd.DataFrame(frame).copy()


def _write_core_tables(output_dir: Path, stats: Any) -> None:
    equity = _coerce_series(_stats_field(stats, "equity", pd.Series(dtype="float64")))
    if not equity.empty:
        equity.to_frame("equity").to_parquet(output_dir / "equity.parquet")

    trades = pd.DataFrame(_stats_field(stats, "trades", pd.DataFrame()))
    trades.to_parquet(output_dir / "trades.parquet", index=False)


def _write_pipeline_weights(output_dir: Path, strategy: Any | None) -> None:
    weights = _pipeline_weights(strategy)
    if weights is not None and not weights.empty:
        weights.to_frame("weight").to_parquet(output_dir / "weights.parquet")


def _pipeline_weights(strategy: Any | None) -> pd.Series | None:
    result = _first_attr(
        strategy,
        ("research_result", "research_result_", "pipeline_result", "pipeline_result_"),
    )
    if result is None:
        return None
    weights = getattr(result, "weights", None)
    if weights is None:
        return None
    return pd.Series(weights, dtype="float64", name="weight")


def _coerce_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.copy()
    return pd.Series(values)


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
