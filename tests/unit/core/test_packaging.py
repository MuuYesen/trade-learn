from __future__ import annotations

import subprocess
import sys
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
        "websocket-client",
        "pynesys-pynecore",
        "causal-learn",
        "numba",
    )

    assert not any(
        dep.startswith(prefix) for dep in dependencies for prefix in optional_runtime_prefixes
    )
    assert any(dep.startswith("opentdx") for dep in extras["research"])
    # TradingView transport is maintained in-tree and uses websocket-client.
    assert any(dep.startswith("websocket-client") for dep in extras["research"])
    assert any(dep.startswith("pynesys-pynecore") for dep in extras["research"])
    assert "TA-Lib==0.8.3" in dependencies
    assert any(dep.startswith("causal-learn") for dep in extras["research"])
    assert any(dep.startswith("numba") for dep in extras["research"])


def test_default_dependencies_exclude_research_and_export_stacks() -> None:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    base = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]
    heavy = ("scipy", "scikit-learn", "pyarrow", "pandas-ta-classic", "bokeh", "xlsxwriter")

    assert not any(dep.lower().startswith(heavy) for dep in base)
    assert any(dep.startswith("scikit-learn") for dep in extras["research"])
    assert any(dep.startswith("scipy") for dep in extras["research"])
    assert any(dep.startswith("pyarrow") for dep in extras["research"])
    assert not any("pandas-ta" in dep.lower() for dep in base + extras["research"])
    assert any(dep.startswith("bokeh") for dep in extras["research"])
    assert any(dep.startswith("xlsxwriter") for dep in extras["research"])


def test_core_facades_import_without_optional_research_and_report_packages() -> None:
    script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        optional = {'scipy', 'sklearn', 'pyarrow', 'pandas_ta_classic',
                    'bokeh', 'xlsxwriter', 'jinja2', 'typer', 'yaml',
                    'requests', 'certifi', 'urllib3', 'idna'}
        if fullname.split('.')[0] in optional:
            raise ImportError(f'optional package imported by core: {fullname}')

sys.meta_path.insert(0, BlockOptional())
import tradelearn.engine
import tradelearn.lite
import tradelearn.cli
from tradelearn.data import TushareProvider
import pandas as pd
provider = TushareProvider(transport=lambda *args: pd.DataFrame({"value": [1]}))
assert provider._call("fixture", {}, "value").value.tolist() == [1]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cibuildwheel_smoke_test_imports_rust_extension() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "scripts/check_preview_wheel.py" in pyproject
    checks = (ROOT / "scripts/check_preview_wheel.py").read_text()
    assert "check_order_lifecycle_wheel.py" in checks
    assert "check_public_runtime_wheel.py" in checks
    smoke = (PYPROJECT.parent / "scripts/check_order_lifecycle_wheel.py").read_text()
    assert "tradelearn_rust_version" in smoke
    assert "cancel_order" in smoke
    assert "configure_order" in smoke


def test_release_workflow_builds_cross_platform_artifacts_and_publishes_with_oidc() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch" in workflow
    assert "tags:" in workflow
    assert "v*" in workflow
    assert "ubuntu-latest" in workflow
    assert "macos-26" in workflow
    assert '"os":"macos-26","cibw-archs":"arm64"' in workflow
    assert "windows-latest" in workflow
    assert "CIBW_ARCHS: ${{ matrix.cibw-archs }}" in workflow
    assert "python -m build --sdist" in workflow
    assert "python -m cibuildwheel --output-dir wheelhouse" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
