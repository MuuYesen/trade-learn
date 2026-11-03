"""Compact exact strategy recovery facts, written only before proven raw GC.

Not a second cloud writer. The trading callback never opens this database.
"""
import json
import os
from pathlib import Path
import sqlite3
import time
import base64
import hashlib
import zlib

KINDS={'order_context','order_filled','order_fill_voided','account_observation'}


class RecoveryIndex:
    def __init__(self,settings,*,retention_seconds=None,max_bytes=None):
        if retention_seconds is not None and (type(retention_seconds) is not int or retention_seconds<=0):raise ValueError('explicit positive recovery retention required')
        if max_bytes is not None and (type(max_bytes) is not int or not 0<max_bytes<=64*1024*1024):raise ValueError('bounded recovery budget required')
        self.max_bytes=max_bytes
        self.retention_seconds=retention_seconds
        self.settings=settings
        self.path=Path(settings.state_dir)/'recovery.sqlite'
        self.binding=json.dumps([settings.source,settings.strategy_id,settings.account_id,
                                 settings.framework_strategy_id,settings.framework_account_id],separators=(',',':'))

    @staticmethod
    def _account_order(fact):
        return (fact['observed_ts_ns'],int(fact['record_id'].rsplit(':',1)[1]))

    @staticmethod
    def _compact(fact):
        kind=fact.get('kind');fields=fact.get('fields',{})
        if kind=='order_context' and fields.get('context_type')!='submit_reference':return None
        if kind=='account_observation' and fields.get('scope')!='framework_account_event':return None
        return fact if kind in KINDS else None

    @staticmethod
    def _pack(fact):
        raw=json.dumps(fact,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>96*1024:raise ValueError('recovery fact exceeds source line bound')
        encoded=dict(kind=fact['kind'],observed_ts_ns=fact.get('observed_ts_ns'),record_id=fact['record_id'],
            _recovery_zlib_v1=dict(raw_length=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(zlib.compress(raw,1)).decode()))
        return json.dumps(encoded,sort_keys=True,separators=(',',':'),allow_nan=False)

    @staticmethod
    def _unpack(body):
        value=json.loads(body)
        if '_recovery_zlib_v1' not in value:return value
        envelope=value['_recovery_zlib_v1'];length=envelope['raw_length']
        if type(length) is not int or not 0<length<=96*1024:raise ValueError('invalid recovery length')
        compressed=base64.b64decode(envelope['data'],validate=True);decoder=zlib.decompressobj()
        raw=decoder.decompress(compressed,96*1024+1)
        if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or len(raw)!=length
                or hashlib.sha256(raw).hexdigest()!=envelope['sha256']):raise ValueError('invalid recovery encoding')
        fact=json.loads(raw)
        if any(fact.get(k)!=value.get(k) for k in ('kind','record_id','observed_ts_ns')):raise ValueError('recovery envelope binding changed')
        return fact

    def _maintain(self,db):
        # facts is a derived restart index, never the raw business archive.
        if db.execute('PRAGMA user_version').fetchone()[0]<2:
            latest=None
            for ordinal,body in db.execute('SELECT ordinal,body FROM facts ORDER BY ordinal'):
                fact=self._compact(self._unpack(body))
                if fact is None:db.execute('DELETE FROM facts WHERE ordinal=?',(ordinal,));continue
                if fact['kind']=='account_observation':
                    if latest is None or self._account_order(fact)>latest[0]:
                        if latest is not None:db.execute('DELETE FROM facts WHERE ordinal=?',(latest[1],))
                        latest=(self._account_order(fact),ordinal)
                    else:db.execute('DELETE FROM facts WHERE ordinal=?',(ordinal,));continue
                db.execute('UPDATE facts SET body=? WHERE ordinal=?',(self._pack(fact),ordinal))
            db.execute('PRAGMA user_version=2')
        if self.retention_seconds is not None:
            cutoff=time.time_ns()-self.retention_seconds*1_000_000_000
            db.execute("DELETE FROM facts WHERE json_extract(body,'$.observed_ts_ns')<? AND json_extract(body,'$.kind')!='account_observation'",(cutoff,))

    def _open(self):
        if self.path.is_symlink():raise ValueError('unsafe strategy recovery index')
        db=sqlite3.connect(self.path)
        os.chmod(self.path,0o600);db.execute('PRAGMA synchronous=FULL')
        db.executescript('CREATE TABLE IF NOT EXISTS binding(value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS facts(ordinal INTEGER PRIMARY KEY AUTOINCREMENT,record_id TEXT UNIQUE NOT NULL,body TEXT NOT NULL);')
        with db:
            row=db.execute('SELECT value FROM binding').fetchone()
            if row is None:db.execute('INSERT INTO binding VALUES(?)',(self.binding,))
            elif row[0]!=self.binding:db.close();raise ValueError('strategy recovery binding changed')
        with db:self._maintain(db)
        return db

    def preserve(self,path):
        db=self._open()
        try:
            with db,Path(path).open('rb') as stream:
                total=db.execute('SELECT coalesce(sum(length(CAST(body AS BLOB))),0) FROM facts').fetchone()[0]
                count=db.execute('SELECT count(*) FROM facts').fetchone()[0]
                budget=min(self.settings.max_disk_bytes if self.max_bytes is None else self.max_bytes,64*1024*1024)
                while raw:=stream.readline(96*1024+1):
                    if not raw.endswith(b'\n'):
                        # Only the final unterminated tail is archive_only. A later
                        # LF means a malformed/oversize complete record, never ignored.
                        while rest:=stream.read(65536):
                            if b'\n' in rest:raise ValueError('invalid complete recovery source line')
                        break
                    if len(raw)>96*1024:raise ValueError('invalid recovery source line')
                    fact=json.loads(raw)
                    if any(fact.get(k)!=getattr(self.settings,k) for k in ('source','strategy_id','account_id')):
                        raise ValueError('strategy recovery source identity changed')
                    if fact.get('kind') not in KINDS:continue
                    if fact.get('kind')=='account_observation' and fact.get('fields',{}).get('scope')!='framework_account_event':continue
                    fact=self._compact(fact)
                    if fact is None:continue
                    if (self.retention_seconds is not None and fact['kind']!='account_observation'
                            and fact['observed_ts_ns']<time.time_ns()-self.retention_seconds*1_000_000_000):continue
                    if fact['kind']=='account_observation':
                        previous=db.execute("SELECT record_id,body FROM facts WHERE json_extract(body,'$.kind')='account_observation' LIMIT 1").fetchone()
                        if previous and previous[0]!=fact['record_id']:
                            if self._account_order(self._unpack(previous[1]))>=self._account_order(fact):continue
                            total-=len(previous[1].encode('utf-8'));count-=1
                            db.execute('DELETE FROM facts WHERE record_id=?',(previous[0],))
                    body=self._pack(fact)
                    old=db.execute('SELECT body FROM facts WHERE record_id=?',(fact['record_id'],)).fetchone()
                    if old is not None and self._unpack(old[0])!=fact:raise ValueError('strategy recovery fact conflict')
                    if old is None:
                        total+=len(body.encode('utf-8'));count+=1
                        if total>budget or count>self.settings.max_event_history*2:
                            raise BufferError('strategy recovery identity/UTF8 budget exhausted; retain raw')
                        db.execute('INSERT INTO facts(record_id,body) VALUES(?,?)',(fact['record_id'],body))
                if db.execute('SELECT count(*) FROM facts').fetchone()[0]>self.settings.max_event_history*2:
                    raise BufferError('strategy recovery identity capacity exhausted')
        finally:db.close()
        from .filesystem import sync_committed_database
        sync_committed_database(self.path)

    def records(self):
        if not self.path.exists():return
        db=self._open()
        try:
            for row in db.execute('SELECT body FROM facts ORDER BY ordinal'):
                yield json.dumps(self._unpack(row[0]),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        finally:db.close()
