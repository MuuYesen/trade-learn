"""Minimal local research CLI; optional components are loaded only when used."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
from pathlib import Path

from tradelearn import __version__
from tradelearn.core.config import load_config
from tradelearn.core.errors import ConfigurationError

# Inspect installed distribution metadata without importing optional runtimes,
# connecting to providers, or reading credentials.
COMPONENTS = (
    "numpy",
    "pandas",
    "TA-Lib",
    "jupyterlab",
    "ipywidgets",
    "jupyter-ai-lite",
    "mlflow",
    "duckdb",
    "scikit-learn",
    "Riskfolio-Lib",
    "optuna",
    "gplearn",
    "causal-learn",
    "pynesys-pynecore",
    "KunQuant",
    "tvkit",
    "nautilus_trader",
    "tl-store",
)


def _doctor(config_path: Path | None) -> int:
    config = load_config(project_dir=Path.cwd(), config_path=config_path)
    print(f"python={platform.python_version()}")
    print(f"trade-learn={__version__}")
    print("configuration=valid")
    print(
        "mlflow_tracking="
        + ("local" if config.mlflow_tracking_uri.startswith(("sqlite:", "file:")) else "explicit")
    )
    missing_base = False
    for package in COMPONENTS:
        try:
            value = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing_base |= package in {"numpy", "pandas", "TA-Lib"}
            value = (
                "not installed (optional)"
                if package not in {"numpy", "pandas", "TA-Lib"}
                else "MISSING (base requirement)"
            )
        print(f"{package}={value}")
    print("tradelearn doctor " + ("failed (missing base requirement)" if missing_base else "ok"))
    return 1 if missing_base else 0


def _new(name: str, directory: Path) -> None:
    if name in {".", "..", ""} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("project name must be a single directory name")
    project = directory / name
    project.mkdir(parents=True, exist_ok=False)
    notebooks = project / "notebooks"
    notebooks.mkdir()
    _write_skill(project)
    (project / "strategy.py").write_text(_starter_strategy(), encoding="utf-8")
    for filename, title in _starter_notebooks().items():
        (notebooks / filename).write_text(_starter_notebook(title), encoding="utf-8")
    (project / "README.md").write_text(
        "# " + name + "\n\nstrategy.py defines an example Strategy; "
        "import it into Cerebro with your own Bars.\n"
        "Start `tradelearn lab --notebook-dir .` from this directory. No YAML is required.\n"
        "The research Skill is written to .agents/skills (read by Codex and "
        "other agent tools) and mirrored at _agents/skills for AI Lite and "
        "Jupyter, which hide dotted paths. Run `tradelearn skills --refresh` "
        "after upgrading trade-learn to rewrite it for the installed version.\n"
        "MLflow uses local sqlite:///mlflow.db by default; no server is started.\n",
        encoding="utf-8",
    )
    print(f"Created tradelearn project: {project}")


_VERSION_LINE = re.compile(r"Target package: trade-learn [^.]*(?:\.[^.]*)*?\. Check the installed version")

# Skill placement. The real directory is dotted `.agents/skills`, which Codex
# and other agent tools read. Jupyter and AI Lite hide dotted paths, so `_agents`
# mirrors it; a symlink keeps one source of truth, and platforms that refuse
# symlinks fall back to a copy that `--refresh` rewrites. `.claude/skills` holds
# the per-skill link Claude Code reads.
SKILL_NAME = "tradelearn-research"
_REAL_SKILL_PARENT = (".agents", "skills")
_MIRROR_DIRS = ("_agents",)


def _link_or_copy(link: Path, target: Path, *, relative_to: Path) -> str:
    """Point `link` at `target`, copying when the platform refuses symlinks."""
    import os
    import shutil

    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(os.path.relpath(target, relative_to), target_is_directory=target.is_dir())
        return "linked"
    except (OSError, NotImplementedError):
        if target.is_dir():
            shutil.copytree(target, link)
        else:
            shutil.copy2(target, link)
        return "copied"


def _write_skill(project: Path) -> Path:
    """Write SKILL.md for the installed version and expose it to every tool."""
    real = project.joinpath(*_REAL_SKILL_PARENT, SKILL_NAME)
    real.mkdir(parents=True, exist_ok=True)
    skill_file = real / "SKILL.md"
    skill_file.write_text(_starter_skill(), encoding="utf-8")
    for mirror in _MIRROR_DIRS:
        _link_or_copy(project / mirror, project / _REAL_SKILL_PARENT[0], relative_to=project)
    claude = project / ".claude" / "skills" / SKILL_NAME
    _link_or_copy(claude, real, relative_to=claude.parent)
    return skill_file


def _skills(directory: Path, refresh: bool) -> int:
    project = directory.resolve()
    if not project.is_dir():
        raise ValueError(f"not a directory: {project}")
    skill_file = project.joinpath(*_REAL_SKILL_PARENT, SKILL_NAME, "SKILL.md")
    if refresh:
        written = _write_skill(project)
        print(f"skill={written}")
        print(f"written_for=trade-learn {__version__}")
        return 0
    if not skill_file.is_file():
        print(f"skill=not found under {project}")
        print("run tradelearn skills --refresh to write it")
        return 1
    first = skill_file.read_text(encoding="utf-8").splitlines()
    stamped = next((ln for ln in first if ln.startswith("Target package: trade-learn ")), "")
    recorded = stamped.split()[3].rstrip(".") if stamped else "unknown"
    print(f"skill={skill_file}")
    print(f"written_for=trade-learn {recorded}")
    print(f"installed=trade-learn {__version__}")
    print("status=" + ("current" if recorded == __version__ else "stale (run --refresh)"))
    return 0 if recorded == __version__ else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="trade-learn local research CLI")
    parser.add_argument("--version", action="version", version=f"tradelearn {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="Check configuration and all installed components")
    doctor.add_argument("--config", type=Path)
    new = commands.add_parser("new", help="Create a minimal project and version-matched Skills")
    new.add_argument("name")
    new.add_argument("--directory", "-d", type=Path, default=Path("."))
    skills = commands.add_parser(
        "skills", help="Show or rewrite the project research Skill for the installed version"
    )
    skills.add_argument("--directory", "-d", type=Path, default=Path("."))
    skills.add_argument("--refresh", action="store_true")
    lab = commands.add_parser("lab", help="Start only JupyterLab")
    lab.add_argument("--config", type=Path)
    lab.add_argument("--host", default="127.0.0.1")
    lab.add_argument("--port", type=int, default=8888)
    lab.add_argument("--notebook-dir", type=Path, default=Path("."))
    lab.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            raise SystemExit(_doctor(args.config))
        elif args.command == "new":
            _new(args.name, args.directory)
        elif args.command == "skills":
            raise SystemExit(_skills(args.directory, args.refresh))
        elif args.command == "lab":
            from tradelearn.lab import build_lab_plan, check_lab_dependencies, start_lab_stack

            missing = check_lab_dependencies()
            if missing:
                parser.error("Missing " + ", ".join(missing) + "; install trade-learn[research]")
            config = load_config(project_dir=args.notebook_dir, config_path=args.config)
            plan = build_lab_plan(
                config,
                host=args.host,
                port=args.port,
                notebook_dir=args.notebook_dir,
                no_browser=args.no_browser,
            )
            raise SystemExit(start_lab_stack(plan))
        else:
            parser.print_help()
    except (ConfigurationError, ValueError, OSError) as exc:
        parser.error(str(exc))


def _starter_strategy() -> str:
    return (
        "import tradelearn.engine as bt\n\n\n"
        "class DemoStrategy(bt.Strategy):\n"
        '    params = (("size", 1),)\n\n'
        "    def next(self) -> None:\n"
        "        if not self.position:\n"
        "            self.buy(size=self.p.size)\n"
    )


def _starter_skill() -> str:
    """Return the published research Skill, stamped with the installed version.

    The body lives in skill_template.md so the file shipped in the wheel, the
    copy published for download at .agents/skills/tradelearn-research, and the
    copy written into a new project are all the same text.
    """
    template = (Path(__file__).parent / "skill_template.md").read_text(encoding="utf-8")
    return _VERSION_LINE.sub(
        f"Target package: trade-learn {__version__}. Check the installed version", template
    )


def _starter_notebooks() -> dict[str, str]:
    return {
        "01_explore.ipynb": "Explore data",
        "02_factor.ipynb": "Factor research",
        "03_backtest.ipynb": "Backtest workflow",
    }


def _starter_notebook(title: str) -> str:
    return (
        "{\n"
        '  "cells": [\n'
        "    {\n"
        '      "cell_type": "markdown",\n'
        '      "metadata": {},\n'
        '      "source": ["# ' + title + '\\n"]\n'
        "    }\n"
        "  ],\n"
        '  "metadata": {\n'
        '    "kernelspec": {\n'
        '      "display_name": "Python 3",\n'
        '      "language": "python",\n'
        '      "name": "python3"\n'
        "    },\n"
        '    "language_info": {"name": "python"}\n'
        "  },\n"
        '  "nbformat": 4,\n'
        '  "nbformat_minor": 5\n'
        "}\n"
    )
