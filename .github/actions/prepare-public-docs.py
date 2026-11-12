"""Build a public documentation view without publishing private workspace references."""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import subprocess
import tarfile
import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
REPOSITORY = "https://github.com/MuuYesen/trade-learn"


def rewrite_links(text: str, document: Path, root: Path, revision: str) -> str:
    """Keep docs links; expose tracked source links and retain private citation labels."""
    root = root.resolve()
    docs = root / "docs"

    def replacement(match):
        label, target = match.groups()
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or target.startswith("#"):
            return match.group(0)
        raw_path = unquote(parts.path)
        normalized = raw_path.replace("../", "").lstrip("/")
        if normalized.startswith("docs/系统规范/"):
            title = Path(normalized).stem
            anchor = "#" + parts.fragment if parts.fragment else ""
            return f"{label}（工作区内部规范：`{title}{anchor}`）"
        if normalized.startswith("docs/参考资料/paseo-session-records/"):
            return f"{label}（工作区验收记录，未作为公开文档附件发布）"
        path = (document.parent / raw_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(
                f"unrecognized external workspace link in {document.relative_to(root)}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing public link in {document.relative_to(root)}: {target}"
            )
        if path.is_relative_to(docs):
            return match.group(0)
        public_path = quote(path.relative_to(root).as_posix(), safe="/")
        fragment = "#" + parts.fragment if parts.fragment else ""
        return f"[{label}]({REPOSITORY}/blob/{revision}/{public_path}{fragment})"

    return LINK.sub(replacement, text)


def preserve_unicode_anchors(docs: Path) -> dict:
    """Add named aliases for referenced Unicode headings, preserving existing TOC IDs."""
    documents = {p.resolve(): p.read_text(encoding="utf-8") for p in docs.rglob("*.md")}
    wanted = {}
    for document, text in documents.items():
        for match in LINK.finditer(text):
            parts = urlsplit(match.group(2))
            fragment = unquote(parts.fragment)
            if parts.scheme or parts.netloc or not any(ord(c) > 127 for c in fragment):
                continue
            target = (document.parent / unquote(parts.path)).resolve() if parts.path else document
            if target in documents:
                wanted.setdefault(target, set()).add(fragment)

    def normalized(value):
        value = unicodedata.normalize("NFKC", html.unescape(value)).lower()
        value = re.sub(r"[^\w\s-]", "", value)
        return re.sub(r"[-\s]+", "-", value).strip("-")

    added = {}
    for document, fragments in wanted.items():
        text = documents[document]
        explicit = set(re.findall(r"\bid=[\"']([^\"']+)[\"']", text))
        explicit.update(re.findall(r"\{\s*#([^\s}]+)", text))
        lines = text.splitlines(keepends=True)
        headings = []
        fence = None
        for index, line in enumerate(lines):
            boundary = re.match(r"^\s*(`{3,}|~{3,})", line)
            if boundary:
                char = boundary.group(1)[0]
                fence = char if fence is None else (None if char == fence else fence)
                continue
            heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
            if fence is None and heading:
                headings.append((index, normalized(heading.group(1))))
        insertions = {}
        for fragment in sorted(fragments - explicit):
            matches = [index for index, title in headings if title == normalized(fragment)]
            if len(matches) != 1:
                raise ValueError(
                    "unresolved Unicode fragment in "
                    + str(document.relative_to(docs))
                    + ": "
                    + fragment
                )
            insertions.setdefault(matches[0], []).append(fragment)
        for index in sorted(insertions, reverse=True):
            aliases = "".join(
                '<a id="' + html.escape(fragment, quote=True) + '"></a>\n\n'
                for fragment in insertions[index]
            )
            lines.insert(index, aliases)
        if insertions:
            document.write_text("".join(lines), encoding="utf-8")
            added[str(document.relative_to(docs))] = [
                f for group in insertions.values() for f in group
            ]
    return added


def stage(root: Path, destination: Path) -> dict:
    """Stage exactly tracked public source; do not edit the workspace or copy its history."""
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("documentation staging must be outside the source checkout")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("exact public revision required")
    public_refs = subprocess.check_output(
        ["git", "ls-remote", "--heads", "--tags", REPOSITORY + ".git"],
        cwd=root,
        text=True,
        timeout=45,
    )
    if revision not in {line.split()[0] for line in public_refs.splitlines()}:
        raise ValueError("only an exact already-public branch/tag revision may be staged")
    body = subprocess.check_output(["git", "archive", revision], cwd=root)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        archive.extractall(destination, filter="data")
    changed = []
    for document in sorted((destination / "docs").rglob("*.md")):
        original = document.read_text(encoding="utf-8")
        public = rewrite_links(original, document, destination, revision)
        if public != original:
            document.write_text(public, encoding="utf-8")
            changed.append(str(document.relative_to(destination)))
    anchor_aliases = preserve_unicode_anchors(destination / "docs")
    result = {
        "anchor_aliases": anchor_aliases,
        "source_revision": revision,
        "public_ref_verified": True,
        "changed_documents": changed,
        "private_rules_or_records_copied": False,
        "workspace_modified": False,
    }
    (destination / "public-docs-staging.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    """Prepare one checked staging tree for the existing strict MkDocs pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.root, args.destination)))


if __name__ == "__main__":
    main()
