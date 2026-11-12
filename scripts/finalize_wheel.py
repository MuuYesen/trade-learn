"""Normalize Cargo SBOM source paths without changing native bytes or wheel tags."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
from copy import copy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def _normalize(value):
    """Retain dependency/license facts while replacing local Cargo source URLs."""
    if isinstance(value, dict):
        return {key: _normalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize(child) for child in value]
    if isinstance(value, str):
        return re.sub(r"path\+file://[^#]*#", "path+file:///source/native#", value)
    return value


def finalize(source: Path, destination: Path) -> dict:
    """Write a checked wheel with identical native bytes and a valid RECORD."""
    if source.name != destination.name:
        raise ValueError("wheel name/version/platform tags must remain unchanged")
    with ZipFile(source) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel members")
        if any(name.endswith(("RECORD.jws", "RECORD.p7s")) for name in names):
            raise ValueError("signed wheels cannot be rewritten")
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("wheel requires one RECORD")
        record = records[0]
        infos = {info.filename: copy(info) for info in archive.infolist()}
        members = {name: archive.read(name) for name in names if name != record}
    native = {
        name: hashlib.sha256(body).hexdigest()
        for name, body in members.items()
        if name.endswith((".so", ".pyd", ".dylib"))
    }
    for name, body in list(members.items()):
        if name.endswith(".cyclonedx.json"):
            members[name] = (json.dumps(_normalize(json.loads(body)), indent=2) + "\n").encode()
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, body in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(body)])
    writer.writerow([record, "", ""])
    members[record] = rows.getvalue().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name, body in sorted(members.items()):
                archive.writestr(infos[name], body, compress_type=ZIP_DEFLATED)
        with ZipFile(temporary) as archive:
            if any(
                hashlib.sha256(archive.read(name)).hexdigest() != digest
                for name, digest in native.items()
            ):
                raise ValueError("native bytes changed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "native_unchanged": native,
    }


def main(argv=None) -> int:
    """Finalize one reviewed compiler-produced wheel; never publish it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args(argv)
    if args.directory is not None:
        if args.source is not None or args.destination is not None:
            parser.error("choose directory or source/destination")
        wheels = sorted(args.directory.glob("*.whl"))
        if not wheels:
            parser.error("directory contains no wheels")
        print(json.dumps([finalize(path, path) for path in wheels], indent=2))
    else:
        if args.source is None or args.destination is None:
            parser.error("source and destination are required")
        print(json.dumps(finalize(args.source, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
