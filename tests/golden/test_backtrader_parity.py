from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import tomllib

from scripts.build_golden import run_expected_job

ROOT = Path(__file__).resolve().parents[2]


def _write_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0],
            "high": [11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0],
            "low": [9.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "close": [10.5, 11.5, 12.5, 13.5, 12.5, 11.5, 10.5],
            "volume": [1000.0] * 7,
        },
        index=pd.date_range("2026-01-01", periods=7, freq="D", tz="UTC"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _dataset(symbol: str) -> dict[str, str]:
    return {
        "symbol": symbol,
        "engine": "tv",
        "exchange": "NASDAQ",
        "start": "2020-01-01",
        "end": "2024-12-31",
        "freq": "1d",
    }


def _trade_signature(payload: dict[str, object]) -> list[tuple[str, float, float, bool, bool]]:
    return [
        (
            str(trade["datetime"]),
            float(trade["size"]),
            float(trade["price"]),
            bool(trade["isopen"]),
            bool(trade["isclosed"]),
        )
        for trade in payload["trades"]
    ]


def test_oracle_group_includes_backtrader() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    oracle = pyproject["dependency-groups"]["oracle"]

    assert "backtrader" in oracle


@pytest.mark.parametrize(
    "strategy",
    [
        "sma_cross",
        "rsi_oversold",
        "bollinger_breakout",
        "macd_cross",
        "tdx_kdj",
        "supertrend_tv",
        "pairs_trading",
        "equal_weight",
        "alpha101_ml",
        "momentum_portfolio",
    ],
)
def test_backtrader_oracle_matches_tradelearn_proxy_strategy(
    tmp_path: Path,
    strategy: str,
) -> None:
    pytest.importorskip("backtrader")
    datasets_root = tmp_path / "datasets"
    parquet = datasets_root / "tv" / "GOOG_2020-01-01_2024-12-31_1d.parquet"
    _write_parquet(parquet)
    out = tmp_path / f"{strategy}.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_backtrader_oracle.py",
            "--strategy",
            strategy,
            "--parquet",
            str(parquet),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    backtrader_payload = json.loads(out.read_text(encoding="utf-8"))
    tradelearn_payload = run_expected_job(strategy, _dataset("GOOG"), datasets_root)

    assert backtrader_payload["source_engine"] == "backtrader"
    assert backtrader_payload["strategy"] == strategy
    assert _trade_signature(backtrader_payload) == _trade_signature(tradelearn_payload)
    assert (
        backtrader_payload["summary"]["final_cash"]
        == pytest.approx(tradelearn_payload["summary"]["final_cash"], rel=1e-4)
    )
    assert (
        backtrader_payload["summary"]["final_value"]
        == pytest.approx(tradelearn_payload["summary"]["final_value"], rel=1e-4)
    )
