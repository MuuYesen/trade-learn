"""Preserve the exact Cargo recorder test output in public Actions annotations."""

import subprocess
import sys

command = [
    "cargo",
    "test",
    "--locked",
    "-p",
    "leafquant-external-log-core",
    "--lib",
    "--test",
    "platform_recorder",
    "--",
    "--nocapture",
]
try:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300
    )
except subprocess.TimeoutExpired as error:
    text = str(error)
    print("::error::" + text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))
    raise SystemExit(1) from error
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
if result.returncode:
    text = (
        (result.stdout + "\n" + result.stderr)
        .replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )
    print("::error::" + text)
raise SystemExit(result.returncode)
