"""Create one explicit CI environment and publish actionable bootstrap failures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def annotation(level: str, title: str, message: str) -> None:
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{escaped}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--probe-default", action="store_true")
    args = parser.parse_args()
    temporary = Path(os.environ["RUNNER_TEMP"])
    suffix = "-".join(
        os.environ.get(k, "local") for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_JOB")
    )
    target = temporary / ("tradelearn-ci-" + suffix)
    receipt = {
        "requested_python": args.python,
        "explicit_environment": str(target),
        "default_environment_existed": Path(".venv").exists(),
        "commands": [],
    }
    commands = [["uv", "venv"]] if args.probe_default else []
    commands.append(["uv", "venv", "--python", args.python, str(target)])
    exit_code = 1
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        receipt["commands"].append(
            {
                "command": command,
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            probe = args.probe_default and index == 0
            annotation(
                "warning" if probe else "error",
                "CI virtual environment bootstrap",
                result.stderr[-6000:] or result.stdout[-6000:],
            )
            if not probe:
                break
        if index == len(commands) - 1 and result.returncode == 0:
            with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
                stream.write(f"UV_PROJECT_ENVIRONMENT={target}\nVIRTUAL_ENV={target}\n")
            exit_code = 0
    (temporary / "ci-environment-diagnostic.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
