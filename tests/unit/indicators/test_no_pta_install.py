"""Import acceptance when every PTA import is actively refused."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "backend,function",
    [
        ("talib", "SMA"),
        ("tdx", "MA"),
        pytest.param(
            "tv",
            "sma",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 11), reason="Fixed PyneCore requires Python 3.11+"
            ),
        ),
    ],
)
def test_runtime_backends_import_without_pta_and_reject_old_native_api(backend, function):
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
from tradelearn.indicators import BACKEND
assert BACKEND.FUNCTION.supports_streaming
assert not hasattr(tradelearn,'pta')
"""
    code = code.replace("BACKEND", backend).replace("FUNCTION", function)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_tv_dependency_rejects_without_legacy_batch_fallback():
    code = """
import importlib.abc, sys
class DenyTV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pynecore' or fullname.startswith('pynecore.'):
            raise ModuleNotFoundError('deliberately absent PyneCore', name=fullname)
sys.meta_path.insert(0, DenyTV())
try:
    from tradelearn.indicators import tv
except ImportError:
    print('TV dependency absence explicitly rejected')
else:
    raise AssertionError('TV silently substituted a legacy/batch backend')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TV dependency absence explicitly rejected" in result.stdout
