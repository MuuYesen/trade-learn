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


def test_new_exposes_one_skill_at_every_agent_tool_location(tmp_path):
    """Codex reads .agents/skills, Claude Code .claude/skills, Jupyter _agents/skills."""
    assert invoke("new", "demo", "-d", str(tmp_path), block_optional=True).returncode == 0
    root = tmp_path / "demo"
    real = root / ".agents/skills/tradelearn-research/SKILL.md"
    assert real.is_file() and not real.is_symlink()
    for mirror in ("_agents/skills/tradelearn-research", ".claude/skills/tradelearn-research"):
        assert (root / mirror / "SKILL.md").read_text() == real.read_text()
    bodies = {
        p.resolve()
        for p in root.rglob("SKILL.md")
        if "tradelearn-research" in str(p)
    }
    assert bodies == {real.resolve()}, "the Skill must have a single source of truth"


def test_skills_reports_stale_after_upgrade_and_refresh_rewrites_it(tmp_path):
    from tradelearn import __version__

    assert invoke("new", "demo", "-d", str(tmp_path), block_optional=True).returncode == 0
    root = tmp_path / "demo"
    real = root / ".agents/skills/tradelearn-research/SKILL.md"

    current = invoke("skills", "-d", str(root), block_optional=True)
    assert current.returncode == 0 and "status=current" in current.stdout

    real.write_text(
        real.read_text().replace(
            f"Target package: trade-learn {__version__}", "Target package: trade-learn 0.0.1"
        )
    )
    stale = invoke("skills", "-d", str(root), block_optional=True)
    assert stale.returncode == 1 and "stale" in stale.stdout

    assert invoke("skills", "-d", str(root), "--refresh", block_optional=True).returncode == 0
    assert f"Target package: trade-learn {__version__}" in real.read_text()
    assert (root / "_agents/skills/tradelearn-research/SKILL.md").read_text() == real.read_text()


def test_skills_without_a_project_reports_missing(tmp_path):
    missing = invoke("skills", "-d", str(tmp_path), block_optional=True)
    assert missing.returncode == 1 and "not found" in missing.stdout


def test_published_skill_matches_the_one_shipped_in_the_package():
    """The downloadable copy and the packaged template must not drift apart."""
    from tradelearn import __version__

    repo = Path(__file__).resolve().parents[2]
    packaged = repo / "tradelearn/cli/skill_template.md"
    published = repo / ".agents/skills/tradelearn-research/SKILL.md"
    assert published.read_text() == packaged.read_text()
    stamp = f"Target package: trade-learn {__version__}."
    assert stamp in published.read_text(), "bump the published Skill when the version changes"
