from __future__ import annotations

from examples.migration import MIGRATION_CHECKPOINTS, run_migration_smoke


def test_migration_smoke_covers_core_migration_checkpoints() -> None:
    result = run_migration_smoke()

    assert set(result["checkpoints"]) == {
        checkpoint.identifier for checkpoint in MIGRATION_CHECKPOINTS
    }
    assert result["backtrader_import"] == "tradelearn.compat.backtrader"
    assert result["line_indexing"]["current"] == 3.0
    assert result["line_indexing"]["previous"] == 2.0
    assert result["cerebro"]["fills"] >= 1
    assert result["ml"]["selector"] == "CausalSelector"
    assert result["report"]["reporter"] == "Reporter"
