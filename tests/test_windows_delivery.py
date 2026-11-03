"""Real Windows native recorder and source-only transport; no AWS connection."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest

pytestmark=pytest.mark.skipif(os.name!='nt',reason='real Windows ACL/namespace APIs')


class Queue:
    def __init__(self):self.items=[]
    def send_message(self,**kwargs):
        from leafquant_monitor.sqs_transport import Envelope
        self.items.append(Envelope.from_json(kwargs['MessageBody']))
        return {'MessageId':'local-fixture'}


class Receipts:
    def __init__(self):self.proof=None
    def fetch(self,**kwargs):return self.proof


def proof(envelope):
    manifest=json.loads(envelope.payload)
    return dict(schema_version=1,binding_id='public-test',status='verified',
                manifest_sha256=hashlib.sha256(envelope.payload).hexdigest(),
                **{k:manifest[k] for k in ('producer_id','generation','file_id','revision','length','sha256')})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args,**kwargs):raise AssertionError('network forbidden in Windows delivery fixture')
    monkeypatch.setattr(socket.socket,'connect',deny)


def test_real_recorder_to_mock_queue_requires_complete_receipt_before_gc(tmp_path):
    from leafquant_monitor.logger import MonitorSettings,TradeLogger
    from leafquant_monitor.sender import SegmentSender
    settings=MonitorSettings('PUBLIC','strategy','account',str(tmp_path),
                             receipt_origin='https://proof.invalid',archive_binding_id='public-test')
    logger=TradeLogger(settings)
    fact=logger.record('lifecycle',{'state':'started'})
    logger.close()
    raw=next((tmp_path/'segments').glob('*.log'));original=raw.read_bytes()
    assert json.loads(original)==fact
    queue=Queue();receipts=Receipts();sender=SegmentSender(settings,queue,'queue',receipt_client=receipts)
    try:
        sender.send_once()
        assert raw.exists(),'SQS ACK alone must not retire raw bytes'
        seal=next(e for e in queue.items if e.kind=='source_seal')
        receipts.proof=dict(proof(seal),binding_id='foreign')
        with pytest.raises(ValueError):sender.send_once()
        assert raw.read_bytes()==original
        receipts.proof=proof(seal)
        # The previous rejected receipt set a bounded retry delay; make only
        # this fixture immediately eligible, preserving production scheduling.
        sender.outbox.db.execute('UPDATE receipt_retries SET next_attempt=0')
        sender.send_once();sender.send_once()
        assert not raw.exists()
        assert sender.outbox.db.execute('SELECT verified FROM source_files').fetchone()==(2,)
        assert not list((tmp_path/'segments').glob('*.source.json'))
        assert not list((tmp_path/'segments').glob('*.native-time.json'))
        assert not list((tmp_path/'segments').glob('*.proof-retired'))
        assert sender.maintenance_error() is None
    finally:sender.close()


def test_acl_broad_grant_is_rejected_and_sender_preflight_creates_no_state(tmp_path):
    from leafquant_monitor.filesystem import check_private_path
    from leafquant_monitor.logger import MonitorSettings,TradeLogger
    keys=tmp_path/'keys.json';keys.write_text(json.dumps({'access_key_id':'fake','secret_access_key':'fake'}))
    config=tmp_path/'sender.json';config.write_text(json.dumps({'region':'test','queue_url':'queue','credentials_file':str(keys)}))
    check_private_path(config);check_private_path(keys)
    result=subprocess.run(['icacls',str(config),'/grant','*S-1-1-0:(R)'],capture_output=True,timeout=20)
    assert result.returncode==0,'fixture must establish a real broad Windows DACL'
    with pytest.raises(ValueError):check_private_path(config)
    state=tmp_path/'must-not-be-created'
    settings=MonitorSettings('PUBLIC','strategy','account',str(state),sender_config=str(config),
                             receipt_origin='https://proof.invalid',archive_binding_id='public-test')
    with pytest.raises(ValueError):TradeLogger(settings)
    assert not state.exists()


def test_proof_retirement_restarts_after_delete_failure_without_replaying_raw(tmp_path,monkeypatch):
    from leafquant_monitor.source_file_sender import SourceFileSender
    from leafquant_monitor.filesystem import check_private_path
    raw=tmp_path/'original.log';raw.write_bytes(b'original\n');check_private_path(raw)
    queue=Queue();database=tmp_path/'outbox.sqlite'
    def create():return SourceFileSender(database,queue,'queue','PUBLIC','strategy','account','public-jsonl-v1',archive_binding_id='public-test')
    sender=create();anchor=dict(schema_version=1,start_ts_ns=1760000000000000000,end_ts_ns=None,source_schema='native-log-file.v1')
    sender.capture(raw,'original',native_time_anchor=anchor,sealed=True);sender.send_once()
    receipt=proof(next(e for e in queue.items if e.kind=='source_seal'))
    unlink=Path.unlink
    def fail(path,*args,**kwargs):
        if path.name.endswith('.proof-retired'):raise OSError('synthetic deletion failure')
        return unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail)
    with pytest.raises(OSError):sender.collect('original',receipt)
    assert not raw.exists()
    retired=tmp_path/'.original.log.proof-retired'
    assert retired.read_bytes()==b'original\n'
    assert sender.db.execute('SELECT verified FROM source_files').fetchone()==(2,)
    sender.close();monkeypatch.setattr(Path,'unlink',unlink)
    sender=create()
    try:
        sender.poll_receipts(Receipts())
        assert not retired.exists()
        assert not raw.exists()
        assert sender.send_once()==0
    finally:sender.close()


def test_binary_source_preserves_crlf_and_non_utf8_exactly(tmp_path):
    from leafquant_monitor.source_file_sender import SourceFileSender
    queue=Queue();raw=tmp_path/'exact.bin';original=b'first\r\n\x00\xfflast\r\n';raw.write_bytes(original)
    sender=SourceFileSender(tmp_path/'outbox.sqlite',queue,'queue','PUBLIC','strategy','account','binary-test',archive_binding_id='public-test',mode='binary')
    try:
        anchor=dict(schema_version=1,start_ts_ns=1760000000000000000,end_ts_ns=None,source_schema='native-log-file.v1')
        sender.capture(raw,'exact',native_time_anchor=anchor,sealed=True);sender.send_once()
        assert b''.join(e.payload for e in queue.items if e.kind=='source_chunk')==original
        seal=next(e for e in queue.items if e.kind=='source_seal')
        assert json.loads(seal.payload)['sha256']==hashlib.sha256(original).hexdigest()
        sender.collect('exact',proof(seal));assert not raw.exists()
    finally:sender.close()


def test_native_publisher_can_replace_sidecar_while_python_reader_pins_old_inode(tmp_path):
    from leafquant_monitor.logger import MonitorSettings,TradeLogger
    from leafquant_monitor.filesystem import open_read_fd
    logger=TradeLogger(MonitorSettings('PUBLIC','strategy','account',str(tmp_path)))
    fd=None
    try:
        logger.record('lifecycle',{'state':'first'});logger.flush()
        side=next((tmp_path/'segments').glob('*.log.source.json'))
        fd=open_read_fd(side)
        old=json.loads(os.read(fd,4096))
        logger.record('lifecycle',{'state':'second'});logger.flush()
        current=json.loads(side.read_bytes())
        assert current['publication_sequence']>old['publication_sequence']
        assert current['published_length']>old['published_length']
        state=json.loads(logger.recorder.status())
        assert state['durable']==2 and state['error'] is None
    finally:
        if fd is not None:os.close(fd)
        logger.close()
