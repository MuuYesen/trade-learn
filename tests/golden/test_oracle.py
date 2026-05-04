from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import tomllib

from scripts import build_golden

ROOT = Path(__file__).resolve().parents[2]
RETIRED_PROVIDER_NAMES = ("pytd" + "x2", "moot" + "dx")


def test_reference_oracle_layout_exists() -> None:
    reference = ROOT / "reference" / "tradelearn_1x"

    assert reference.is_dir()
    assert (reference / "query" / "query.py").is_file()


def test_check_oracle_cli_reports_readiness() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_oracle.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "oracle=ok" in result.stdout
    assert "provider:" in result.stdout


def test_check_oracle_cli_prints_oracle_group_hint_when_provider_missing() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_oracle.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    if "provider:" in result.stdout and "=missing" in result.stdout:
        assert "hint=uv sync --group oracle --extra dev" in result.stdout


def test_pyproject_has_oracle_dependency_group() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    oracle = pyproject["dependency-groups"]["oracle"]

    assert any(dep.startswith("opentdx") for dep in dependencies)
    assert any(dep.startswith("tradingview-datafeed") for dep in dependencies)
    assert not any(dep.startswith(RETIRED_PROVIDER_NAMES[0]) for dep in dependencies)
    assert not any(dep.startswith("yfinance") for dep in oracle)
    assert any(dep.startswith("opentdx") for dep in oracle)
    assert not any(dep.startswith(RETIRED_PROVIDER_NAMES[1]) for dep in oracle)
    assert not any("git+" in dep for dep in dependencies)
    assert not any("git+" in dep for dep in oracle)


def test_current_provider_source_uses_opentdx_not_retired_names() -> None:
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "scripts" / "build_golden.py",
        ROOT / "scripts" / "check_oracle.py",
        ROOT / "tradelearn" / "data" / "providers.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "opentdx" in combined
    assert RETIRED_PROVIDER_NAMES[0] not in combined
    assert RETIRED_PROVIDER_NAMES[1] not in combined


def test_build_golden_default_failure_writes_no_expected_json(tmp_path: Path) -> None:
    existing_dataset = (
        ROOT / "tests" / "golden" / "datasets" / "tv" / "EXISTING_TEST.parquet"
    )
    existing_dataset.parent.mkdir(parents=True, exist_ok=True)
    existing_dataset.write_bytes(b"existing")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/build_golden.py",
                "--version",
                "1.x",
                "--out",
                str(tmp_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert not list(tmp_path.glob("*.json"))
        assert existing_dataset.read_bytes() == b"existing"
    finally:
        existing_dataset.unlink(missing_ok=True)


def test_build_golden_dry_run_is_read_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "jobs=100" in result.stdout
    assert not any(tmp_path.iterdir())


def test_build_golden_datasets_only_uses_provider_stubs(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_load_reference_query(*, allow_provider_stubs: bool = False):
        calls.append(allow_provider_stubs)
        query = Mock()
        query.history_ohlc.side_effect = build_golden.GoldenDataError("network unavailable")
        return query

    monkeypatch.setattr(build_golden, "load_reference_query", fake_load_reference_query)
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": True, "tv": True})

    result = build_golden.main(
        ["--version", "1.x", "--out", str(tmp_path), "--datasets-only"]
    )

    assert result == 2
    assert calls == [True]


def test_build_golden_datasets_only_attempts_every_dataset(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    manifest = {
        "datasets": [
            {
                "symbol": "GOOG",
                "exchange": "NASDAQ",
                "engine": "tv",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
            {
                "symbol": "SZ.000001",
                "engine": "tdx",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
        ],
        "strategies": [],
    }
    query = Mock()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1),
            "code": ["000001"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
        }
    )
    query.history_ohlc.side_effect = [build_golden.GoldenDataError("bad tv symbol"), frame]

    monkeypatch.setattr(build_golden, "load_manifest", lambda: manifest)
    monkeypatch.setattr(build_golden, "load_reference_query", lambda **_: query)
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": True, "tv": True})

    result = build_golden.main(
        ["--version", "1.x", "--out", str(tmp_path), "--datasets-only"]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert query.history_ohlc.call_count == 2
    assert query.history_ohlc.call_args_list[0].kwargs["exchange"] == "NASDAQ"
    assert "dataset=tv:NASDAQ:GOOG status=failed reason=bad tv symbol" in captured.out
    assert "dataset=tdx:SZ.000001 status=ok" in captured.out
    assert "datasets=1/2" in captured.out


def test_build_golden_datasets_only_can_generate_tv_subset_without_tdx(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    manifest = {
        "datasets": [
            {
                "symbol": "GOOG",
                "exchange": "NASDAQ",
                "engine": "tv",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
            {
                "symbol": "000001",
                "engine": "tdx",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
        ],
        "strategies": [],
    }
    query = Mock()
    query.history_ohlc.return_value = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1),
            "code": ["GOOG"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
        }
    )

    monkeypatch.setattr(build_golden, "load_manifest", lambda: manifest)
    monkeypatch.setattr(build_golden, "load_reference_query", lambda **_: query)
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": False, "tv": True})
    datasets_root = tmp_path / "datasets"

    result = build_golden.main(
        [
            "--version",
            "1.x",
            "--out",
            str(tmp_path),
            "--datasets-only",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert query.history_ohlc.call_count == 1
    assert query.history_ohlc.call_args.kwargs["engine"] == "tv"
    assert "dataset=tv:NASDAQ:GOOG status=ok" in captured.out
    assert "datasets=1/1" in captured.out
    assert (
        datasets_root / "tv" / "GOOG_2020-01-01_2020-01-02_1d.parquet"
    ).exists()


def test_build_golden_expected_can_generate_tv_subset_from_strategy_adapter(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    manifest = {
        "datasets": [
            {
                "symbol": "GOOG",
                "exchange": "NASDAQ",
                "engine": "tv",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
        ],
        "strategies": [{"name": "sma_cross"}],
    }
    datasets_root = tmp_path / "datasets"
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )
    frame.to_parquet(tv_dir / "GOOG_2020-01-01_2020-01-02_1d.parquet")

    monkeypatch.setattr(build_golden, "load_manifest", lambda: manifest)

    result = build_golden.main(
        [
            "--version",
            "1.x",
            "--out",
            str(tmp_path / "expected"),
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
        ]
    )

    payload = json.loads((tmp_path / "expected" / "sma_cross__GOOG.json").read_text())
    captured = capsys.readouterr()
    assert result == 0
    assert payload["strategy"] == "sma_cross"
    assert payload["dataset"] == "GOOG"
    assert payload["engine"] == "tv"
    assert payload["summary"]["bars"] == 4
    assert isinstance(payload["trades"], list)
    assert "expected=1/1" in captured.out


def test_build_golden_backtrader_oracle_generates_supported_tv_subset(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    strategies = [
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
    ]
    manifest = {
        "datasets": [
            {
                "symbol": "GOOG",
                "exchange": "NASDAQ",
                "engine": "tv",
                "start": "2020-01-01",
                "end": "2020-01-02",
                "freq": "1d",
            },
        ],
        "strategies": [{"name": name} for name in strategies],
    }
    datasets_root = tmp_path / "datasets"
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True)
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
    frame.to_parquet(tv_dir / "GOOG_2020-01-01_2020-01-02_1d.parquet")

    monkeypatch.setattr(build_golden, "load_manifest", lambda: manifest)

    result = build_golden.main(
        [
            "--version",
            "1.x",
            "--out",
            str(tmp_path / "expected"),
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--oracle",
            "backtrader",
        ]
    )

    payload = json.loads((tmp_path / "expected" / "sma_cross__GOOG.json").read_text())
    captured = capsys.readouterr()
    assert result == 0
    assert payload["source_engine"] == "backtrader"
    assert payload["strategy"] == "sma_cross"
    assert payload["dataset"] == "GOOG"
    assert (tmp_path / "expected" / "rsi_oversold__GOOG.json").exists()
    assert (tmp_path / "expected" / "momentum_portfolio__GOOG.json").exists()
    assert "expected=10/10" in captured.out


def test_build_golden_datasets_only_reports_unavailable_opentdx(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": False, "tv": True})

    result = build_golden.main(
        ["--version", "1.x", "--out", str(tmp_path), "--datasets-only"]
    )

    assert result == 2
    assert "dataset provider unavailable: tdx:opentdx.tdxClient" in capsys.readouterr().err


def test_fetch_dataset_reports_opentdx_bridge_error(monkeypatch) -> None:
    """TDX oracle failures preserve the underlying opentdx diagnostics."""
    query = Mock()
    def fail_with_recorded_bridge_error(**_: object) -> None:
        build_golden._LAST_REFERENCE_TDX_ERROR = (
            "opentdx connection not established for 1.2.3.4:7709"
        )
        return None

    query.history_ohlc.side_effect = fail_with_recorded_bridge_error
    dataset = {
        "symbol": "000001",
        "engine": "tdx",
        "start": "2020-01-01",
        "end": "2020-01-02",
        "freq": "1d",
    }
    try:
        build_golden.fetch_dataset(query, dataset)
    except build_golden.GoldenDataError as exc:
        assert "opentdx connection not established for 1.2.3.4:7709" in str(exc)
    else:
        raise AssertionError("expected GoldenDataError")
