"""Use the existing maturin release entry with source paths remapped.

Caller owns the shared heavy slot and verifies available space. No installer,
public uploader, cross-ABI fabrication or official NA modification is included.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_environment(root: Path, environ=None) -> dict[str, str]:
    """Add Cargo path remapping while preserving effective caller Rust flags."""
    env = dict(os.environ if environ is None else environ)
    encoded = env.get("CARGO_ENCODED_RUSTFLAGS")
    flags = encoded.split("\x1f") if encoded else env.get("RUSTFLAGS", "").split()
    home = Path.home()
    mappings = [
        (home, "/build"),
        (Path(env.get("CARGO_HOME", home / ".cargo")).resolve(), "/cargo"),
        (Path(env.get("RUSTUP_HOME", home / ".rustup")).resolve(), "/rust"),
        (root.resolve(), "/source/tradelearn"),
    ]
    flags.extend(f"--remap-path-prefix={source}={target}" for source, target in mappings)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return env


def main(argv=None) -> int:
    """Build one real host-interpreter wheel using locked Rust dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--print-cargo-flags", action="store_true")
    parser.add_argument("--interpreter", default=sys.executable)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.print_cargo_flags:
        print(build_environment(ROOT)["CARGO_ENCODED_RUSTFLAGS"])
        return 0
    if args.out is None:
        parser.error("--out is required for a wheel build")
    command = [
        "maturin",
        "build",
        "--release",
        "--locked",
        "--interpreter",
        args.interpreter,
        "--out",
        str(args.out.resolve()),
    ]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, cwd=ROOT, env=build_environment(ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
