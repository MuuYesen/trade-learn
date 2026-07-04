from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"
RELEASE_WORKFLOW = ROOT / ".github/workflows/release.yml"


def test_cibuildwheel_config_targets_supported_python_and_platforms() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "[tool.cibuildwheel]" in pyproject
    assert 'build = "cp310-* cp311-* cp312-* cp313-* cp314-*"' in pyproject
    assert '"Programming Language :: Python :: 3.13"' in pyproject
    assert '"Programming Language :: Python :: 3.14"' in pyproject
    assert 'skip = "*-musllinux_*"' in pyproject
    assert "[tool.cibuildwheel.linux]" in pyproject
    assert "[tool.cibuildwheel.macos]" in pyproject
    assert "[tool.cibuildwheel.windows]" in pyproject


def test_optional_backends_are_not_required_for_core_install() -> None:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    optional_runtime_prefixes = (
        "opentdx",
        "tradingview-datafeed",
        "TA-Lib",
        "pynesys-pynecore",
        "causal-learn",
        "numba",
    )

    assert not any(
        dep.startswith(prefix)
        for dep in dependencies
        for prefix in optional_runtime_prefixes
    )
    assert any(dep.startswith("opentdx") for dep in extras["tdx"])
    assert any(dep.startswith("tradingview-datafeed") for dep in extras["tv"])
    assert any(dep.startswith("pynesys-pynecore") for dep in extras["tv"])
    assert any(dep.startswith("TA-Lib") for dep in extras["talib"])
    assert any(dep.startswith("causal-learn") for dep in extras["ml"])
    assert any(dep.startswith("numba") for dep in extras["research"])


def test_cibuildwheel_smoke_test_imports_rust_extension() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "import tradelearn._rust as rust" in pyproject
    assert "tradelearn_rust_version" in pyproject


def test_release_workflow_builds_cross_platform_artifacts_and_publishes_with_oidc() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "tags:" in workflow
    assert "v*" in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-14" in workflow
    assert "cibw-archs: arm64" in workflow
    assert "windows-latest" in workflow
    assert "CIBW_ARCHS: ${{ matrix.cibw-archs }}" in workflow
    assert "python -m build --sdist" in workflow
    assert "python -m cibuildwheel --output-dir wheelhouse" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
