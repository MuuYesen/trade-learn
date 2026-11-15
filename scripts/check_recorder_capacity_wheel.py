"""Synthetic installed-wheel capacity recovery check; no source overlay or network."""
import pathlib,tempfile,json,sys,types,importlib.metadata
import tradelearn, leafquant_monitor
from tradelearn import _rust as module
root=pathlib.Path(__file__).resolve().parents[1]
assert not pathlib.Path(tradelearn.__file__).resolve().is_relative_to(root), "installed wheel required"
assert importlib.metadata.version("trade-learn")=="0.3.7rc1"
from leafquant_monitor.logger import TradeLogger
with tempfile.TemporaryDirectory(prefix='ic-retry-sdk-host-') as tmp:
 folder=pathlib.Path(tmp);r=module.Recorder(tmp,8,128,300);logger=TradeLogger.__new__(TradeLogger);logger.recorder=r;logger.sender=None;logger._error=None;logger.settings=types.SimpleNamespace(capacity=8)
 line=lambda n:json.dumps({'n':n,'body':'x'*80},separators=(',',':'))
 assert r.append(line(1));assert logger.flush()['durable']==1
 assert r.append(line(2))
 try:logger.flush()
 except RuntimeError:pass
 else:raise AssertionError('missing capacity failure')
 old=json.loads(r.status());assert old['accepted']==2 and old['durable']==1
 logger._sequence=0;logger.run_id=old['run_id'];logger._monotonic_ns=lambda:1
 logger.settings=types.SimpleNamespace(capacity=8,source='SYNTHETIC',strategy_id='test',account_id='test')
 try:logger.record('heartbeat',{})
 except RuntimeError:pass
 else:raise AssertionError('missing append rejection')
 assert logger._error=='storage: recorder disk limit exceeded', 'Known capacity rejection must retain its precise recovery category' 
 log=next(folder.glob('*.log'));assert log.read_text()==line(1)+'\n';log.unlink();log.with_suffix('.native-time.json').unlink()
 result=logger.flush();assert result['accepted']==result['durable']==2 and result['error'] is None
 assert logger.health()['error'] is None, 'SDK wrapper must stop overriding recovered native state with the prior capacity error'
 assert logger.health_summary()['recorder_code']=='OK'
 # Success must not clear any unrelated prior failure.
 logger._error='OSError';logger.flush();assert logger.health()['error']=='OSError'
 r.close();assert next(folder.glob('*.log')).read_text()==line(2)+'\n'
 print(json.dumps({'actual_pyo3_recorder':True,'sdk_flush_capacity_recovery':True,'unrelated_error_retained':True,'original_binding_source':'rust/tradelearn-rust/src/logging.rs','synthetic_only':True}))
