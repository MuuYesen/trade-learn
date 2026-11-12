"""Final user-approved CLI; no service wrappers or optional-dependency guesses."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def invoke(*args, cwd=None, block_optional=False):
    code = """
import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'typer', 'yaml', 'mlflow', 'jupyterlab'}:
   raise ImportError('optional package blocked')
"""
    if block_optional:
        code += "sys.meta_path.insert(0, Block())\n"
    code += "from tradelearn.cli import main\nmain()\n"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        capture_output=True,
        text=True,
    )


def test_basic_help_version_doctor_need_no_typer_yaml_or_optional_imports(tmp_path):
    for args in [("--help",), ("--version",), ("doctor",)]:
        result = invoke(*args, cwd=tmp_path, block_optional=True)
        assert result.returncode == 0, result.stderr
    result = invoke("doctor", cwd=tmp_path, block_optional=True)
    assert "python=" in result.stdout and "trade-learn=" in result.stdout
    assert "jupyter-ai-lite=" in result.stdout and "tl-store=" in result.stdout
    assert "mlflow_tracking=local" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ("mcp", "--help"),
        ("mlflow", "--help"),
        ("doctor", "--lab"),
        ("doctor", "--mlflow"),
        ("lab", "--with-mlflow"),
        ("lab", "--mlflow-port", "5050"),
    ],
)
def test_retired_commands_and_flags_are_rejected(args):
    result = invoke(*args)
    assert result.returncode == 2, result.stdout


def test_new_project_without_yaml_and_skill_visible_to_default_contents(tmp_path):
    result = invoke("new", "demo", "-d", str(tmp_path), block_optional=True)
    assert result.returncode == 0, result.stderr
    root = tmp_path / "demo"
    assert not (root / ".tradelearn/config.yaml").exists()
    skill = root / "_agents/skills/tradelearn-research/SKILL.md"
    assert skill.is_file()
    assert (
        "ResearchRun" in skill.read_text() and "tradelearn.integrations.mlflow" in skill.read_text()
    )
    compile((root / "strategy.py").read_text(), str(root / "strategy.py"), "exec")
    assert invoke("new", "demo", "-d", str(tmp_path)).returncode != 0


def test_mcp_public_implementation_is_retired():
    spec = importlib.util.find_spec("tradelearn.mcp")
    assert spec is None or spec.loader is None
    assert not any(Path(__file__).resolve().parents[2].joinpath("tradelearn/mcp").glob("*.py"))


def test_default_tracking_has_no_unstarted_http_server():
    from tradelearn.core.config import TradelearnConfig

    assert TradelearnConfig().mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_doctor_optional_absence_succeeds_but_missing_base_is_failure(monkeypatch, capsys):
    import tradelearn.cli as cli

    def version(name):
        if name in {"numpy", "pandas", "TA-Lib"}:
            return "fixture"
        raise cli.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.importlib.metadata, "version", version)
    assert cli._doctor(None) == 0
    assert "not installed (optional)" in capsys.readouterr().out

    def missing(name):
        raise cli.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.importlib.metadata, "version", missing)
    assert cli._doctor(None) == 1
    assert "MISSING (base requirement)" in capsys.readouterr().out
