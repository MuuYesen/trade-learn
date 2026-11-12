"""Public links resolve without exposing private workspace documents or hiding typos."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "public_docs_links", ROOT / ".github/actions/prepare-public-docs.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def project(tmp_path):
    (tmp_path / "docs/guides").mkdir(parents=True)
    (tmp_path / "docs/target.md").write_text("# Target\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/demo.py").write_text("VALUE = 1\n")
    return tmp_path / "docs/guides/test.md"


def test_public_code_link_targets_the_exact_source_revision(tmp_path):
    document = project(tmp_path)
    actual = helper.rewrite_links("[demo](../../examples/demo.py)", document, tmp_path, "reviewed")
    assert (
        actual == "[demo](https://github.com/MuuYesen/trade-learn/blob/reviewed/examples/demo.py)"
    )


@pytest.mark.parametrize(
    "link",
    ["[page](../target.md#target)", "[external](https://example.invalid/)", "[anchor](#here)"],
)
def test_existing_docs_and_external_links_are_retained(tmp_path, link):
    assert helper.rewrite_links(link, project(tmp_path), tmp_path, "reviewed") == link


def test_private_norm_is_cited_without_copying_or_linking_private_source(tmp_path):
    actual = helper.rewrite_links(
        "[规范](../../../../../docs/系统规范/交易/TradeLearn框架规范.md#runtime)",
        project(tmp_path),
        tmp_path,
        "reviewed",
    )
    assert "工作区内部规范" in actual and "TradeLearn框架规范#runtime" in actual
    assert "](../../" not in actual


def test_missing_public_link_is_still_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        helper.rewrite_links("[typo](../typo.md)", project(tmp_path), tmp_path, "reviewed")


def test_unrecognized_outside_workspace_link_is_not_silently_hidden(tmp_path):
    with pytest.raises(ValueError):
        helper.rewrite_links(
            "[unknown](../../../../unknown.md)", project(tmp_path), tmp_path, "reviewed"
        )


def test_unicode_fragment_gets_an_alias_without_changing_the_original_heading(tmp_path):
    document = project(tmp_path)
    document.write_text("[orders](../target.md#订单生命周期)\n")
    (tmp_path / "docs/target.md").write_text("# 订单生命周期\n")
    helper.preserve_unicode_anchors(tmp_path / "docs")
    rendered = (tmp_path / "docs/target.md").read_text()
    assert '<a id="订单生命周期"></a>' in rendered
    assert "# 订单生命周期" in rendered


def test_existing_explicit_unicode_anchor_is_not_duplicated(tmp_path):
    document = project(tmp_path)
    document.write_text("[orders](../target.md#订单生命周期)\n")
    (tmp_path / "docs/target.md").write_text('<a id="订单生命周期"></a>\n\n# Orders\n')
    helper.preserve_unicode_anchors(tmp_path / "docs")
    assert (tmp_path / "docs/target.md").read_text().count('id="订单生命周期"') == 1


def test_bad_unicode_fragment_remains_a_real_error(tmp_path):
    document = project(tmp_path)
    document.write_text("[typo](../target.md#不存在)\n")
    with pytest.raises(ValueError, match="unresolved Unicode fragment"):
        helper.preserve_unicode_anchors(tmp_path / "docs")
