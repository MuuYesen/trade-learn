from __future__ import annotations

import math

from examples.quickstart import run_quickstart


def test_quickstart_example_runs_end_to_end() -> None:
    result = run_quickstart()

    assert result["strategy"] == "QuickstartSmaCross"
    assert result["bars"] >= 20
    assert result["fills"] >= 1
    assert math.isfinite(result["final_value"])
    assert math.isfinite(result["return_pct"])
