from __future__ import annotations

from pathlib import Path

import yaml


def test_docs_workflow_deploys_mkdocs_site_with_github_pages_actions() -> None:
    workflow_path = Path(".github/workflows/docs.yml")
    assert workflow_path.exists()

    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["name"] == "Docs"
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["pages"] == "write"
    assert workflow["permissions"]["id-token"] == "write"
    assert "main" in workflow[True]["push"]["branches"]
    assert "v2" in workflow[True]["push"]["branches"]
    build_steps = workflow["jobs"]["build"]["steps"]
    step_uses = [step.get("uses") for step in build_steps if "uses" in step]
    step_runs = [step.get("run") for step in build_steps if "run" in step]

    assert "astral-sh/setup-uv@v5" in step_uses
    assert "actions/configure-pages@v5" in step_uses
    assert "actions/upload-pages-artifact@v3" in step_uses
    assert "uv sync --extra docs" in step_runs
    assert "uv run mkdocs build --strict" in step_runs
    assert workflow["jobs"]["deploy"]["steps"][0]["uses"] == "actions/deploy-pages@v4"
