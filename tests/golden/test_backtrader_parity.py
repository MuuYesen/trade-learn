from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import tomllib

from scripts.build_golden import run_expected_job
from scripts.run_backtrader_oracle import (
    run_backtrader_oracle,
    summarize_backtrader_metrics,
)

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


def _write_bars_contract_parquet(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="D", tz="UTC"),
            "symbol": ["NASDAQ:GOOG"] * 7,
            "open": [10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0],
            "high": [11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0],
            "low": [9.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "close": [10.5, 11.5, 12.5, 13.5, 12.5, 11.5, 10.5],
            "volume": [1000.0] * 7,
        }
    ).set_index(["timestamp", "symbol"])
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


def test_backtrader_oracle_summary_exposes_native_analyzer_metrics(tmp_path: Path) -> None:
    pytest.importorskip("backtrader")
    datasets_root = tmp_path / "datasets"
    parquet = datasets_root / "tv" / "GOOG_2020-01-01_2024-12-31_1d.parquet"
    _write_parquet(parquet)
    out = tmp_path / "sma_cross.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_backtrader_oracle.py",
            "--strategy",
            "sma_cross",
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
    summary = json.loads(out.read_text(encoding="utf-8"))["summary"]

    expected_keys = {
        "annual_return",
        "avg_trade_pct",
        "drawdown_len",
        "expectancy",
        "final_realized_pnl",
        "max_drawdown",
        "max_drawdown_len",
        "peak_value",
        "profit_factor",
        "return_pct",
        "sharpe",
        "sqn",
        "total_trades",
        "win_rate_pct",
    }
    assert expected_keys <= set(summary)
    assert "kelly_criterion" not in summary
    assert "exposure_pct" not in summary


def test_backtrader_oracle_accepts_bars_contract_multiindex(tmp_path: Path) -> None:
    pytest.importorskip("backtrader")
    parquet = tmp_path / "GOOG_2020-01-01_2024-12-31_1d.parquet"
    _write_bars_contract_parquet(parquet)

    payload = run_backtrader_oracle("sma_cross", parquet, dataset="GOOG")

    assert payload["dataset"] == "GOOG"
    assert payload["summary"]["bars"] == 7
    assert payload["summary"]["final_value"] == pytest.approx(99_998.0)


def test_summarize_backtrader_metrics_uses_common_native_summary_shape() -> None:
    equity = pd.Series(
        [100_000.0, 101_000.0, 99_000.0, 102_000.0],
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    trades = [
        {"pnl": 100.0, "value": 1000.0, "isclosed": True},
        {"pnl": -50.0, "value": 1000.0, "isclosed": True},
    ]

    summary = summarize_backtrader_metrics(
        bars=4,
        initial_cash=100_000.0,
        final_cash=102_000.0,
        final_value=102_000.0,
        orders=4,
        fills=4,
        trades=trades,
        equity=equity,
        analyzers={
            "drawdown": {"drawdown": 1.0, "len": 2, "max": {"drawdown": 3.0, "len": 4}},
            "returns": {"rnorm100": 5.0, "rtot": 0.02},
            "sharpe": {"sharperatio": 1.5},
            "sqn": {"sqn": 2.5, "trades": 2},
            "trades": {
                "total": {"closed": 2},
                "won": {"total": 1, "pnl": {"total": 100.0}},
                "lost": {"total": 1, "pnl": {"total": -50.0}},
                "pnl": {"net": {"total": 50.0, "average": 25.0}},
            },
        },
    )

    assert summary["bars"] == 4
    assert summary["return_pct"] == pytest.approx(2.0)
    assert summary["total_trades"] == 2
    assert summary["annual_return"] == 5.0
    assert summary["avg_trade_pct"] == pytest.approx(0.025)
    assert summary["expectancy"] == 25.0
    assert summary["final_realized_pnl"] == 50.0
    assert summary["max_drawdown"] == pytest.approx(0.03)
    assert summary["peak_value"] == 102_000.0
    assert summary["profit_factor"] == 2.0
    assert summary["drawdown_len"] == 2
    assert summary["max_drawdown_len"] == 4
    assert summary["sharpe"] == 1.5
    assert summary["sqn"] == 2.5
    assert summary["win_rate_pct"] == 50.0
    assert "kelly_criterion" not in summary
