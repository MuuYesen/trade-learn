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
    assert "master" in workflow[True]["push"]["branches"]
    assert "main" in workflow[True]["push"]["branches"]
    assert "v2" in workflow[True]["push"]["branches"]
    build_steps = workflow["jobs"]["build"]["steps"]
    step_uses = [step.get("uses") for step in build_steps if "uses" in step]
    step_runs = [step.get("run") for step in build_steps if "run" in step]

    assert "astral-sh/setup-uv@v5" in step_uses
    assert "actions/configure-pages@v5" in step_uses
    assert "actions/upload-pages-artifact@v3" in step_uses
    assert "uv sync --frozen --group docs" in step_runs
    build = next(step for step in build_steps if step.get("name") == "Build site")
    assert "prepare-public-docs.py" in build["run"]
    assert 'cd "$RUNNER_TEMP/trade-learn-public-docs"' in build["run"]
    assert '"$DOCS_INTERPRETER" -m mkdocs build --strict' in build["run"]
    upload = next(
        step for step in build_steps if step.get("uses") == "actions/upload-pages-artifact@v3"
    )
    assert upload["with"]["path"] == "${{ runner.temp }}/trade-learn-public-docs/site"
    assert workflow["jobs"]["deploy"]["steps"][0]["uses"] == "actions/deploy-pages@v4"
