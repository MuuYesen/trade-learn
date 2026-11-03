"""Validate the external Python/native release and successful CI before publishing."""

import argparse
import ast
import json
import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def release_version(root=ROOT):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?", project)
    if not match:
        raise ValueError("release requires major.minor.patch with optional a/b/rc number")
    expected_native = match[1]
    if match[2]:
        expected_native += "-" + {"a": "alpha", "b": "beta", "rc": "rc"}[match[2]] + "." + match[3]
    crate = tomllib.loads((root / "rust/tradelearn-rust/Cargo.toml").read_text())["package"][
        "version"
    ]
    cargo = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    cargo_versions = [row["version"] for row in cargo if row["name"] == "tradelearn-rust"]
    lock = tomllib.loads((root / "uv.lock").read_text())["package"]
    python_versions = [row["version"] for row in lock if row["name"] == "trade-learn"]
    assignments = ast.parse((root / "tradelearn/__init__.py").read_text()).body
    runtime_versions = [
        ast.literal_eval(node.value)
        for node in assignments
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if (
        crate != expected_native
        or cargo_versions != [expected_native]
        or python_versions != [project]
        or runtime_versions != [project]
    ):
        raise ValueError("Python, native crate, Cargo.lock and uv.lock release versions differ")
    return project, bool(match[2])


def successful_ci(repository, sha, token=None):
    query = urllib.parse.urlencode({"head_sha": sha, "status": "completed", "per_page": 100})
    url = f"https://api.github.com/repos/{repository}/actions/workflows/ci.yml/runs?{query}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        runs = json.load(response)["workflow_runs"]
    matching = [
        run
        for run in runs
        if run["head_sha"] == sha
        and run["head_repository"]["full_name"].lower() == repository.lower()
    ]
    if not matching or max(matching, key=lambda run: run["updated_at"])["conclusion"] != "success":
        raise ValueError(
            "publishing requires completed successful CI for this exact repository commit"
        )


def check(
    ref, event, publish_requested, *, root=ROOT, repository="", sha="", ci_check=successful_ci
):
    version, prerelease = release_version(root)
    tag = ref.startswith("refs/tags/")
    publish = tag or publish_requested
    if tag and ref != "refs/tags/v" + version:
        raise ValueError("version tag must exactly match the external package version")
    if publish and not tag:
        raise ValueError("publishing requires the exact version tag; branch dispatch is build-only")
    if publish:
        if event not in ("push", "workflow_dispatch") or not repository or not sha:
            raise ValueError("publishing requires a reviewed repository commit and tag event")
        ci_check(repository, sha, os.environ.get("GITHUB_TOKEN"))
    return {
        "version": version,
        "prerelease": str(prerelease).lower(),
        "publish": str(publish).lower(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--publish", choices=("true", "false"), default="false")
    args = parser.parse_args()
    result = check(
        args.ref,
        args.event,
        args.publish == "true",
        repository=os.environ.get("GITHUB_REPOSITORY", ""),
        sha=os.environ.get("GITHUB_SHA", ""),
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write("".join(f"{name}={value}\n" for name, value in result.items()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
