from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReportContext:
    """Context passed to custom report sections."""

    reporter: Any
    stats: Any
    returns: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame
    benchmark: pd.Series | None
    market_data: pd.DataFrame | None
    research_result: Any | None = None


def build_context(reporter: Any, benchmark: pd.Series | None = None) -> ReportContext:
    """Build a stable custom-section context from a reporter."""
    stats = getattr(reporter, "stats", None)
    return ReportContext(
        reporter=reporter,
        stats=stats,
        returns=pd.Series(reporter._get("returns")),
        trades=pd.DataFrame(reporter._get("trades", default=pd.DataFrame())),
        positions=pd.DataFrame(reporter._get("positions", default=pd.DataFrame())),
        orders=pd.DataFrame(reporter._get("orders", default=pd.DataFrame())),
        benchmark=None if benchmark is None else pd.Series(benchmark),
        market_data=getattr(reporter, "market_data", None),
        research_result=_research_result(stats),
    )


def render_html_sections(
    sections: Sequence[Any] | None,
    context: ReportContext,
) -> str:
    """Render custom HTML sections."""
    rendered: list[str] = []
    for section in sections or ():
        if callable(section) and not hasattr(section, "html"):
            value = section(context)
            if isinstance(value, Mapping):
                value = value.get("html")
        elif hasattr(section, "html"):
            value = section.html(context)
        else:
            continue
        if value:
            rendered.append(str(value))
    return "\n".join(rendered)


def collect_excel_sheets(
    sections: Sequence[Any] | None,
    context: ReportContext,
) -> dict[str, pd.DataFrame]:
    """Collect custom Excel sheets from sections."""
    sheets: dict[str, pd.DataFrame] = {}
    for section in sections or ():
        if callable(section) and not hasattr(section, "excel"):
            value = section(context)
            if isinstance(value, Mapping):
                value = value.get("excel")
        elif hasattr(section, "excel"):
            value = section.excel(context)
        else:
            value = None
        if value is None or value is False:
            continue
        if isinstance(value, pd.DataFrame | pd.Series):
            name = str(getattr(section, "name", "custom"))
            sheets[_excel_sheet_name(name)] = _as_frame(value)
            continue
        if isinstance(value, Mapping):
            for name, table in value.items():
                sheets[_excel_sheet_name(str(name))] = _as_frame(table)
    return sheets


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def _excel_sheet_name(name: str) -> str:
    invalid = "[]:*?/\\"
    cleaned = "".join("_" if char in invalid else char for char in name).strip("'")
    return (cleaned or "custom")[:31]


def _research_result(stats: Any) -> Any | None:
    strategy = getattr(stats, "strategy", None)
    if strategy is not None:
        for name in ("research_result", "research_result_"):
            if hasattr(strategy, name):
                return getattr(strategy, name)
    config = getattr(stats, "config", None)
    if isinstance(config, Mapping):
        return config.get("research")
    return None


__all__ = ["ReportContext"]
