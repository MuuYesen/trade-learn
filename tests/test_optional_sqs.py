"""Explicit optional delivery must fail before touching durable local state."""
import importlib.abc
import sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import pytest
from leafquant_monitor.logger import MonitorSettings, TradeLogger

class NoAws(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('boto3', 'botocore'):
            raise ModuleNotFoundError(name=fullname)


def test_metadata_keeps_aws_out_of_base_and_in_nautilus_extra():
    data=tomllib.loads((Path(__file__).parents[1]/'pyproject.toml').read_text())['project']
    assert not any('boto' in dep for dep in data['dependencies'])
    assert 'boto3==1.41.2' in data['optional-dependencies']['nautilus']
    assert any(dep.startswith('botocore==') for dep in data['optional-dependencies']['nautilus'])


def test_missing_sqs_dependency_rejects_before_state_or_recorder(tmp_path, monkeypatch):
    for name in list(sys.modules):
        if name.split('.')[0] in ('boto3','botocore'):monkeypatch.delitem(sys.modules,name)
    block=NoAws();sys.meta_path.insert(0,block)
    root=tmp_path/'never-created'
    settings=MonitorSettings(source='fixture',strategy_id='s',account_id='a',state_dir=str(root),
        sender_config=str(tmp_path/'not-read.json'),receipt_origin='https://receipt.invalid',archive_binding_id='binding')
    try:
        with pytest.raises(ImportError,match=r'trade-learn\[nautilus\]'):
            TradeLogger(settings)
        assert not root.exists()
    finally:sys.meta_path.remove(block)
