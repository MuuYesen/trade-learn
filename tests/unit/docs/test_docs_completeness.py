from __future__ import annotations

from pathlib import Path

from scripts.check_docs_completeness import check_docs_completeness


def test_docs_completeness_requires_all_spec_pages_in_mkdocs_nav() -> None:
    report = check_docs_completeness(Path("."))

    assert report.ok
    assert report.missing_spec_nav == ()
    assert "specs/ARCHITECTURE.md" in report.spec_nav_paths
    assert "specs/BACKTEST_SPEC.md" in report.spec_nav_paths
    assert "specs/STRATEGY_SPEC.md" in report.spec_nav_paths


def test_docs_completeness_requires_release_pages_and_design_notes() -> None:
    report = check_docs_completeness(Path("."))

    assert report.missing_release_pages == ()
    assert report.missing_internal_nav == ()
    assert "README.md" in report.nav_paths
    assert "PROJECT.md" in report.nav_paths
    assert "RUNBOOK.md" in report.nav_paths
    assert "internal/event-loop.md" in report.nav_paths
    assert "internal/matching-design.md" in report.nav_paths
    assert "internal/portfolio.md" in report.nav_paths


def test_chinese_readme_documents_runtime_boundaries_and_artifact_schema() -> None:
    text = Path("README_zh.md").read_text(encoding="utf-8")

    assert "TDX / MyTT 经典指标已覆盖 30+ 常用函数" in text
    assert "PyneCore common subset" in text
    assert "不是完整 TradingView/Pine 语言实现" in text
    assert "artifact_schema_version=1.0" in text
    assert "artifact_kind=backtest" in text
    assert "破坏性变更必须提升 schema version" in text
    assert "不引入 action-dict / `next_multi()` 这类并列策略模型" in text
