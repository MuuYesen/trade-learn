"""Preserve uv dependency-check failures in public Actions annotations."""

import subprocess
import sys


def main():
    result = subprocess.run(["uv", "pip", "check"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, end="")
    if result.returncode:
        escaped = output.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print("::error::" + escaped)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
