"""The base facade remains usable without the optional execution dependency."""
import subprocess
import sys


def test_store_namespace_does_not_load_native_engine_or_logging():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from tradelearn import engine as bt
assert 'NautilusStore' in dir(bt.stores)
assert 'nautilus_trader' not in sys.modules
assert 'tradelearn_runtime.execution' not in sys.modules
assert 'leafquant_monitor' not in sys.modules
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_nautilus_store_reports_missing_optional_dependency():
    result = subprocess.run([sys.executable, '-c', '''
import builtins
from tradelearn import engine as bt
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'nautilus_trader':
        raise ModuleNotFoundError(name=name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
try:
    bt.stores.NautilusStore
except RuntimeError as error:
    assert 'trade-learn[nautilus]' in str(error) or 'Python 3.13 or 3.14' in str(error)
else:
    raise AssertionError('missing dependency was silently ignored')
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
