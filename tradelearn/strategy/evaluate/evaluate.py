"""Legacy Evaluate facade backed by the v2 report layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tradelearn.report import Reporter


class Evaluate:
    """Compatibility entry point for historical strategy evaluation reports."""

    @staticmethod
    def analysis_report(
        stats: Any,
        baseline: pd.DataFrame,
        filename: str | None = None,
        engine: str = "quantstats",
    ) -> None:
        """Write an HTML report using the v2 Reporter facade.

        The legacy ``pyfolio`` and ``quantstats`` engine names are accepted as
        compatibility aliases; both now route through ``tradelearn.report``.
        """
        if engine not in {"pyfolio", "quantstats"}:
            raise ValueError("engine must be 'pyfolio' or 'quantstats'")
        output = Path(filename or f"./{engine}.html")
        returns = pd.Series(stats._equity_curve.Equity).pct_change().dropna()
        benchmark = pd.Series(baseline.close).pct_change().dropna()
        Reporter(
            {
                "returns": returns,
                "trades": pd.DataFrame(),
                "summary": {"strategy_name": engine},
                "config": {"strategy": engine},
            }
        ).html(output, benchmark=benchmark)
