"""Verify frozen source bytes, wheel RECORD and sdist closure; never run models.

A passing receipt covers artifact contents only. Installed/runtime/platform
acceptance remains a separate gate. Manifests use the existing SOURCE-READY
``files: {relative_path: sha256}`` mapping, including explicit build snapshots.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

PACKAGES = {"tradelearn", "tradelearn_runtime", "leafquant_monitor"}


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"invalid relative artifact path: {name}")
    return path


def read_wheel(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("wheel duplicate members")
        for name in names:
            relative(name)
        members = {name: archive.read(name) for name in names}
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise ValueError("record requires exactly one wheel RECORD")
    record = records[0]
    rows = list(csv.reader(io.StringIO(members[record].decode())))
    if any(len(row) != 3 for row in rows):
        raise ValueError("record malformed row")
    if len(rows) != len(members) or {row[0] for row in rows} != set(members):
        raise ValueError("record member coverage mismatch")
    for name, checksum, size in rows:
        if name == record:
            if checksum or size:
                raise ValueError("record must leave its own digest and size empty")
            continue
        body = members[name]
        encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        if checksum != "sha256=" + encoded or size != str(len(body)):
            raise ValueError(f"record mismatch: {name}")
    return members


def read_sdist(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        roots = {relative(member.name).parts[0] for member in files}
        if len(roots) != 1:
            raise ValueError("sdist must have one root directory")
        members = {}
        for member in files:
            name = "/".join(relative(member.name).parts[1:])
            if not name or name in members:
                raise ValueError("sdist duplicate or invalid member")
            members[name] = archive.extractfile(member).read()
    return members


def verify(root: Path, wheel: Path, sdist: Path, manifests: list[Path]) -> dict:
    """Reject stale/omitted bytes; return no claim of runtime or release readiness."""
    root = root.resolve()
    frozen: dict[str, str] = {}
    manifest_hashes = {}
    for manifest in manifests:
        body = manifest.read_bytes()
        payload = json.loads(body)
        supplied = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(supplied, dict) or not supplied:
            raise ValueError(f"manifest requires nonempty files mapping: {manifest.name}")
        manifest_hashes[str(manifest)] = digest(body)
        for name, checksum in supplied.items():
            path = relative(name)
            if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
                raise ValueError(f"manifest invalid digest: {name}")
            if name in frozen and frozen[name] != checksum:
                raise ValueError(f"manifest conflicting frozen digest: {name}")
            source = root / path
            if not source.resolve().is_relative_to(root):
                raise ValueError(f"source escapes project root: {name}")
            if not source.is_file() or digest(source.read_bytes()) != checksum:
                raise ValueError(f"source mismatch: {name}")
            frozen[name] = checksum
    if not frozen:
        raise ValueError("manifest requires frozen inputs")
    wheel_members = read_wheel(wheel)
    sdist_members = read_sdist(sdist)
    prefixes = [str(Path.home()).encode(), str(root).encode()]
    for kind, members in (("wheel", wheel_members), ("sdist", sdist_members)):
        for name, body in members.items():
            if any(prefix in body for prefix in prefixes):
                raise ValueError(f"{kind} private build path: {name}")
    wheel_files = 0
    for name, checksum in frozen.items():
        if name not in sdist_members or digest(sdist_members[name]) != checksum:
            raise ValueError(f"sdist frozen input missing or mismatched: {name}")
        if relative(name).parts[0] in PACKAGES:
            if name not in wheel_members or digest(wheel_members[name]) != checksum:
                raise ValueError(f"wheel frozen input missing or mismatched: {name}")
            wheel_files += 1
    return {
        "scope": "frozen source bytes, wheel RECORD, sdist closure and private path scan",
        "artifact_closure_verified": True,
        "runtime_verified": False,
        "frozen_files": len(frozen),
        "wheel_frozen_files": wheel_files,
        "wheel_members": len(wheel_members),
        "wheel": str(wheel.resolve()),
        "wheel_sha256": digest(wheel.read_bytes()),
        "sdist": str(sdist.resolve()),
        "sdist_sha256": digest(sdist.read_bytes()),
        "manifest_sha256": manifest_hashes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = verify(args.root, args.wheel, args.sdist, args.manifest)
        code = 0
    except (ValueError, OSError, tarfile.TarError, BadZipFile) as error:
        receipt = {"artifact_closure_verified": False, "error": str(error)}
        code = 1
    content = json.dumps(receipt, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(content)
    print(content, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
