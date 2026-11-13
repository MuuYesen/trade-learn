"""Build-only installed Linux consumer gate; never grants release approval."""
import hashlib,json,os,pathlib,subprocess,sys,tempfile,types,zipfile
root=pathlib.Path(__file__).resolve().parents[1]
wheel=next((root/'consumer-wheels').glob('*.whl'))
import tradelearn,tradelearn._rust,tradelearn.engine as bt
site=pathlib.Path(tradelearn.__file__).parent.parent
assert not pathlib.Path(tradelearn.__file__).is_relative_to(root)
with zipfile.ZipFile(wheel) as z:
    expected={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith(('tradelearn/','tradelearn_runtime/','leafquant_monitor/'))}
def verify():
    for n,h in expected.items():assert hashlib.sha256((site/n).read_bytes()).hexdigest()==h,n
verify()
code="""import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('nautilus_trader','tl_store','boto3','botocore','tradelearn_runtime'):raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from tradelearn import engine as bt
assert 'NautilusStore' in bt.stores.__all__
"""
tests=types.ModuleType('tests');tests.__path__=[str(root/'tests')];sys.modules['tests']=tests
import pytest
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'PYTHONPATH':''})
    config=pathlib.Path(tmp)/'pytest.ini';config.write_text('[pytest]\n')
    result=pytest.main(['-q','--import-mode=importlib','--confcutdir',tmp,'-c',str(config),'-o','pythonpath=',str(root/'tests/unit/engine/test_public_store_entry.py'),str(root/'tests/unit/engine/test_nautilus_store.py'),'-k','(public_store_entry and not namespace_does_not_load) or official_bar_replay_uses_one_runtime','-W','error::pytest.PytestUnraisableExceptionWarning'])
verify()
for name,module in tuple(sys.modules.items()):
    if name.split('.')[0] in ('tradelearn','tradelearn_runtime','leafquant_monitor') and getattr(module,'__file__',None):assert pathlib.Path(module.__file__).is_relative_to(site),(name,module.__file__)
import importlib.metadata as m
receipt={'exit_code':int(result),'installed_package_members_before_after':len(expected),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'native_sha256':hashlib.sha256(pathlib.Path(tradelearn._rust.__file__).read_bytes()).hexdigest(),'platform':sys.platform,'python':sys.version,'versions':{n:m.version(n) for n in ('trade-learn','tl-store','nautilus_trader')},'template_sha256':hashlib.sha256((root/'examples/live/ctrader_monitor.py').read_bytes()).hexdigest(),'source_overlay':False,'public_pypi_release':False,'real_monitor_sender_archive':False,'all_C1_C8_closed':False}
(root/'consumer-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(result)
