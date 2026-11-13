"""Installed CP311 owner ABI gate for the same frozen SDK; no NA/live secrets."""
import hashlib,json,os,pathlib,subprocess,sys,tempfile,zipfile
root=pathlib.Path(__file__).resolve().parents[1]
wheel=next((root/'consumer-wheels').glob('*cp311*.whl'))
import tradelearn,tradelearn._rust,leafquant_monitor
site=pathlib.Path(tradelearn.__file__).parent.parent
assert sys.version_info[:2]==(3,11)
assert not pathlib.Path(tradelearn.__file__).is_relative_to(root)
with zipfile.ZipFile(wheel) as z:
    expected={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith(('tradelearn/','tradelearn_runtime/','leafquant_monitor/'))}
    sdk={n:h for n,h in expected.items() if n.startswith('leafquant_monitor/') and n.endswith('.py')}
def verify():
    for n,h in expected.items():assert hashlib.sha256((site/n).read_bytes()).hexdigest()==h,n
    for n,h in sdk.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
verify()
assert len(sdk)==19
with tempfile.TemporaryDirectory() as tmp:
    result=subprocess.run([sys.executable,str(root/'scripts/native_recorder_smoke.py'),tradelearn._rust.__file__],cwd=tmp,check=True,env={**os.environ,'PYTHONPATH':''})
verify()
import importlib.metadata as metadata
out={'python':sys.version,'platform':sys.platform,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'native_sha256':hashlib.sha256(pathlib.Path(tradelearn._rust.__file__).read_bytes()).hexdigest(),'sdk_hashes':sdk,'installed_members_before_after_equal':len(expected),'source_overlay':False,'native_recorder_smoke':True,'boto3':metadata.version('boto3'),'botocore':metadata.version('botocore'),'NA_execution_tested':False,'public_two_field_store_entry_tested':False,'real_sender_archive_roundtrip':False,'public_pypi_release':False}
(root/'consumer-qualification-311.json').write_text(json.dumps(out,indent=2)+'\n')
