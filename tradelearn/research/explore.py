from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.research.run import tracked


@dataclass(frozen=True)
class DataProfile:
    """Compact profile for a raw research dataset."""

    rows: int
    columns: int
    dtypes: pd.Series
    missing: pd.Series
    missing_rate: pd.Series
    numeric: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly profile payload."""
        return {
            "shape": {"rows": self.rows, "columns": self.columns},
            "dtypes": {str(key): str(value) for key, value in self.dtypes.items()},
            "missing": {str(key): int(value) for key, value in self.missing.items()},
            "missing_rate": {
                str(key): float(value) for key, value in self.missing_rate.items()
            },
            "numeric": _frame_dict(self.numeric),
        }


@tracked(category="explore")
def profile(data: pd.DataFrame) -> DataProfile:
    """Return a compact raw-data profile."""
    frame = pd.DataFrame(data)
    missing = frame.isna().sum()
    numeric = frame.describe().T if not frame.select_dtypes("number").empty else pd.DataFrame()
    return DataProfile(
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        dtypes=frame.dtypes,
        missing=missing,
        missing_rate=missing / len(frame) if len(frame) else missing.astype("float64"),
        numeric=numeric,
    )


@tracked(category="explore")
def report(data: pd.DataFrame, path: str | Path = "explore.html") -> Path:
    """Write a simple HTML raw-data profile report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = profile(data)
    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head><meta charset='utf-8'><title>TradeLearn Data Profile</title></head>",
            "<body>",
            "<h1>TradeLearn Data Profile</h1>",
            f"<p>Rows: {summary.rows} | Columns: {summary.columns}</p>",
            "<h2>Dtypes</h2>",
            summary.dtypes.astype(str).to_frame("dtype").to_html(),
            "<h2>Missing</h2>",
            pd.DataFrame(
                {
                    "missing": summary.missing,
                    "missing_rate": summary.missing_rate,
                }
            ).to_html(),
            "<h2>Numeric Summary</h2>",
            summary.numeric.to_html() if not summary.numeric.empty else "<p>No numeric columns.</p>",
            "</body>",
            "</html>",
        ]
    )
    output.write_text(html, encoding="utf-8")
    return output


def _frame_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row_key, values in frame.to_dict(orient="index").items():
        payload[str(row_key)] = {
            str(key): _json_value(value) for key, value in values.items()
        }
    return payload


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = ["DataProfile", "profile", "report"]
