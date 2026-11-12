"""Build infrastructure must not silently bypass the local heavy owner."""

import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("KunQuant")


def verifier():
    return runpy.run_path(str(Path(__file__).with_name("kunquant_fixture.py")))[
        "verify_build_owner"
    ]


def test_local_build_requires_tl_nonce_and_refuses_foreign_owner(tmp_path):
    verify = verifier()
    path = tmp_path / "owner"
    path.write_text(json.dumps({"owner": "WEB-INTEGRATION", "nonce": "other"}))
    with pytest.raises(RuntimeError):
        verify(owner_path=path)
    path.write_text(
        json.dumps({"owner": "TL-INTEGRATION", "task": "TL-INTEGRATION", "nonce": "own"})
    )
    assert verify(owner_path=path) == "local-heavy"


def test_ci_mode_requires_actual_github_actions_environment(monkeypatch):
    verify = verifier()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(RuntimeError):
        verify(github_actions=True)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    assert verify(github_actions=True) == "github-hosted-isolated-job"
