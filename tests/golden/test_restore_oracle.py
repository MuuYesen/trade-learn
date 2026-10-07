"""The frozen oracle must be reproducible without replacing local changes."""

import hashlib
from pathlib import Path

import pytest

from scripts import restore_oracle


def test_restore_oracle_uses_verified_history(tmp_path: Path) -> None:
    restored = restore_oracle.restore(tmp_path)
    assert len(restored) == 13
    for item in restore_oracle.manifest()["files"]:
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert restore_oracle.restore(tmp_path) == []


def test_restore_oracle_refuses_to_overwrite_modified_file(tmp_path: Path) -> None:
    item = restore_oracle.manifest()["files"][0]
    path = tmp_path / item["path"]
    path.parent.mkdir(parents=True)
    path.write_text("local change")
    with pytest.raises(ValueError, match="modified"):
        restore_oracle.restore(tmp_path)
    assert path.read_text() == "local change"
