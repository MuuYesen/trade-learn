"""Exercise installed native components and expose original Actions failures."""

import runpy
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from tradelearn import _rust


def annotation(text):
    print(
        "::error::" + text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"), flush=True
    )


def run_checked(command, *, cwd=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        annotation(result.stdout + "\n" + result.stderr)
        raise SystemExit(result.returncode)


for name in ("check_order_lifecycle_wheel.py", "check_public_runtime_wheel.py"):
    try:
        runpy.run_path(str(Path(__file__).with_name(name)), run_name="__main__")
    except Exception:
        annotation(name + "\n" + traceback.format_exc())
        raise

# Each native lifecycle runs in a new process, using this wheel's real binary.
run_checked(
    [sys.executable, str(Path(__file__).with_name("native_recorder_smoke.py")), _rust.__file__]
)
assert Path(_rust.__file__).is_relative_to(Path(sys.prefix)), _rust.__file__
tests = [Path(__file__).resolve().parents[1] / "tests/test_filesystem_identity.py"]
if sys.platform == "win32":
    tests.append(Path(__file__).resolve().parents[1] / "tests/test_windows_delivery.py")
with tempfile.TemporaryDirectory() as directory:
    config = Path(directory) / "pytest.ini"
    config.write_text("[pytest]\n")
    code = """import pathlib,sys,leafquant_monitor,pytest
sdk = pathlib.Path(leafquant_monitor.__file__)
assert sdk.is_relative_to(pathlib.Path(sys.prefix)), sdk
print('Installed SDK:', leafquant_monitor.__file__)
raise SystemExit(pytest.main(sys.argv[1:]))
"""
    run_checked(
        [
            sys.executable,
            "-c",
            code,
            "-q",
            "--import-mode=importlib",
            "--confcutdir",
            directory,
            "-c",
            str(config),
            "-o",
            "pythonpath=",
            *[str(test) for test in tests],
        ],
        cwd=directory,
    )
