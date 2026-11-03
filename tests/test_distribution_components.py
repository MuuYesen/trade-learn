"""Public distribution includes real component sources, not checkout forwards."""
from pathlib import Path
import ast
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_public_distribution_contains_logger_recorder_and_neutral_bridge():
    for relative in ('leafquant_monitor/logger.py', 'leafquant_monitor/tradelearn.py',
                     'tradelearn_runtime/broker.py', 'tradelearn_runtime/execution.py'):
        assert (ROOT / relative).is_file(), relative
        ast.parse((ROOT / relative).read_text())
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert 'nautilus' in metadata['project']['optional-dependencies']
    assert not any('nautilus' in dep.lower() for dep in metadata['project']['dependencies'])


def test_bridge_rejects_unsupported_python_without_importing_nautilus():
    import subprocess, sys
    code = "import sys; sys.version_info=(3,12,0); import tradelearn_runtime.execution"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires Python 3.13 or 3.14' in result.stderr


def test_bridge_missing_dependency_is_explicit():
    import subprocess, sys
    code = """import sys, importlib.abc
sys.version_info=(3,13,0)
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='nautilus_trader': raise ModuleNotFoundError(name=name)
sys.meta_path.insert(0,Block())
import tradelearn_runtime.execution
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires the verified trade-learn[nautilus] dependency' in result.stderr
