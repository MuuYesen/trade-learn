"""Run unchanged wheel checks and expose their original traceback in Actions."""

import runpy
import traceback
from pathlib import Path

for name in ("check_order_lifecycle_wheel.py", "check_public_runtime_wheel.py"):
    try:
        runpy.run_path(str(Path(__file__).with_name(name)), run_name="__main__")
    except Exception:
        message = (
            (name + "\n" + traceback.format_exc())
            .replace("%", "%25")
            .replace("\r", "%0D")
            .replace("\n", "%0A")
        )
        print("::error::" + message, flush=True)
        raise
