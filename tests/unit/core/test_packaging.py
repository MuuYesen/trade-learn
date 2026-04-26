from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"


def test_cibuildwheel_config_targets_supported_python_and_platforms() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "[tool.cibuildwheel]" in pyproject
    assert 'build = "cp310-* cp311-* cp312-*"' in pyproject
    assert 'skip = "*-musllinux_* pp*"' in pyproject
    assert "[tool.cibuildwheel.linux]" in pyproject
    assert "[tool.cibuildwheel.macos]" in pyproject
    assert "[tool.cibuildwheel.windows]" in pyproject


def test_cibuildwheel_smoke_test_imports_rust_extension() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert "import tradelearn._rust as rust" in pyproject
    assert "tradelearn_rust_version" in pyproject
