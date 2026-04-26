from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from scripts import check_release_golden


def _readiness(ok: bool = True) -> dict[str, Any]:
    return {
        "ok": ok,
        "summary": {
            "datasets_ready": 5 if ok else 4,
            "datasets_total": 5,
            "expected_ready": 50 if ok else 49,
            "expected_total": 50,
            "strategies_ready": 10,
            "strategies_total": 10,
        },
        "blockers": {"datasets": [] if ok else [{"reason": "missing real parquet"}]},
    }


def _comparison(ok: bool = True) -> dict[str, Any]:
    return {
        "ok": ok,
        "summary": {"compared": 50 if ok else 49, "failed": 0 if ok else 1},
        "failures": [] if ok else [{"reason": "trades differ"}],
    }


def test_release_golden_gate_requires_readiness_compare_and_pytest(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        check_release_golden,
        "build_readiness_report",
        lambda **_: _readiness(),
    )
    monkeypatch.setattr(
        check_release_golden,
        "compare_golden",
        lambda **_: _comparison(),
    )

    def fake_runner(args: list[str]):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="44 passed", stderr="")

    payload = check_release_golden.build_report(pytest_runner=fake_runner)

    assert payload["ok"] is True
    assert payload["readiness"]["summary"]["expected_ready"] == 50
    assert payload["comparison"]["summary"] == {"compared": 50, "failed": 0}
    assert payload["pytest"]["returncode"] == 0
    assert calls == [["tests/golden", "-q"]]


def test_release_golden_gate_fails_when_compare_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        check_release_golden,
        "build_readiness_report",
        lambda **_: _readiness(),
    )
    monkeypatch.setattr(
        check_release_golden,
        "compare_golden",
        lambda **_: _comparison(ok=False),
    )

    payload = check_release_golden.build_report(
        pytest_runner=lambda _: SimpleNamespace(returncode=0, stdout="44 passed", stderr="")
    )

    assert payload["ok"] is False
    assert payload["comparison"]["failures"][0]["reason"] == "trades differ"


def test_release_golden_cli_json_can_skip_pytest(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_release_golden,
        "build_readiness_report",
        lambda **_: _readiness(),
    )
    monkeypatch.setattr(
        check_release_golden,
        "compare_golden",
        lambda **_: _comparison(),
    )

    exit_code = check_release_golden.main(["--json", "--skip-pytest"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["pytest"]["skipped"] is True
