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
            "missing_rate": {str(key): float(value) for key, value in self.missing_rate.items()},
            "numeric": _frame_dict(self.numeric),
        }

    def to_analysis(self) -> dict[str, Any]:
        """Expose completed profile tables to the common renderer."""
        return {
            "title": "TradeLearn Data Profile",
            "tables": {
                "numeric": self.numeric.rename_axis("column").reset_index(),
                "schema": pd.DataFrame(
                    {
                        "dtype": self.dtypes,
                        "missing": self.missing,
                        "missing_rate": self.missing_rate,
                    }
                )
                .rename_axis("column")
                .reset_index(),
            },
            "metadata": {"rows": self.rows, "columns": self.columns},
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
    from tradelearn.report import Reporter

    return Reporter.from_analysis(profile(data).to_analysis()).report(path)


def _frame_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row_key, values in frame.to_dict(orient="index").items():
        payload[str(row_key)] = {str(key): _json_value(value) for key, value in values.items()}
    return payload


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = ["DataProfile", "profile", "report"]
