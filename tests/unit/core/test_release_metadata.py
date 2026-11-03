"""Publishing is gated on one external version, exact tag and successful CI."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "release_check", ROOT / "scripts/check_release_metadata.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def repo(tmp_path, version="0.2.9", native="0.2.9"):
    (tmp_path / "rust/tradelearn-rust").mkdir(parents=True)
    (tmp_path / "tradelearn").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="' + version + '"\n')
    (tmp_path / "rust/tradelearn-rust/Cargo.toml").write_text(
        '[package]\nversion="' + native + '"\n'
    )
    (tmp_path / "Cargo.lock").write_text(
        '[[package]]\nname="tradelearn-rust"\nversion="'
        + native
        + '"\n[[package]]\nname="logging-core"\nversion="0.1.0"\n'
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname="trade-learn"\nversion="' + version + '"\n'
    )
    (tmp_path / "tradelearn/__init__.py").write_text('__version__="' + version + '"\n')
    return tmp_path


def test_build_only_branch_cannot_accidentally_publish():
    assert (
        guard.check("refs/heads/publish/preview-runtime-20261103", "push", False)["publish"]
        == "false"
    )
    with pytest.raises(ValueError, match="exact version tag"):
        guard.check("refs/heads/master", "workflow_dispatch", True)


def test_wrong_tag_and_native_version_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="exactly match"):
        guard.check("refs/tags/v0.2.8", "push", False)
    with pytest.raises(ValueError, match="versions differ"):
        guard.release_version(repo(tmp_path, native="0.2.8"))


def test_release_requires_exact_commit_ci_and_allows_internal_core_version():
    seen = []
    result = guard.check(
        "refs/tags/v0.2.9",
        "push",
        False,
        repository="owner/repo",
        sha="reviewed",
        ci_check=lambda repository, sha, token: seen.append((repository, sha)),
    )
    assert seen == [("owner/repo", "reviewed")]
    assert result == {"version": "0.2.9", "prerelease": "false", "publish": "true"}


def test_rc_release_sets_prerelease_and_checks_normalized_native_version(tmp_path):
    root = repo(tmp_path, version="0.2.10rc1", native="0.2.10-rc.1")
    assert (
        guard.check(
            "refs/tags/v0.2.10rc1",
            "push",
            False,
            root=root,
            repository="owner/repo",
            sha="reviewed",
            ci_check=lambda *a: None,
        )["prerelease"]
        == "true"
    )
