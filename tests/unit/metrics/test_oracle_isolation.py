"""Legacy oracle imports must not affect unrelated test collection."""
import subprocess
import sys
from pathlib import Path


def test_metrics_oracle_does_not_install_global_provider_stubs() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", """
import runpy, sys
before_path = list(sys.path)
names = ('IPython', 'IPython.display', 'yfinance', 'pandas_datareader', 'pandas_datareader.data')
before = {name: sys.modules.get(name) for name in names}
runpy.run_path('tests/consistency/test_metrics.py')
assert sys.path == before_path
for name in names:
    module = sys.modules.get(name)
    assert module is before[name] or (module is not None and module.__spec__ is not None), name
"""],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
