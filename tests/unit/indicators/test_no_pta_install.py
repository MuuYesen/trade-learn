"""Import acceptance when every PTA import is actively refused."""

import os
import subprocess
import sys
from pathlib import Path


def test_runtime_backends_import_without_pta_and_reject_old_native_api():
    import tradelearn

    root = Path(tradelearn.__file__).resolve().parent.parent
    code = """
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith('pandas_ta'):
        raise ImportError('PTA unavailable in this acceptance environment')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import tradelearn
from tradelearn.indicators import talib,tv,tdx
assert talib.SMA.supports_streaming and tv.sma.supports_streaming and tdx.MA.supports_streaming
assert not hasattr(tradelearn,'pta')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
