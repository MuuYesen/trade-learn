"""Bounded exact replay of unreceipted, retained SDK log source files."""
import hashlib
import json
import math
import sqlite3
from pathlib import Path
import time

from .filesystem import original_identity
from .source_file_sender import SourceFileSender,canonical
from .source_file_contract import validate_source_manifest,validate_native_time_anchor,MAX_CHUNK_BYTES
from .sqs_transport import SourceEnvelope



class RetryingSourceFileSender(SourceFileSender):
    """Reuse the outbox and unchanged identity; no receipt means no deletion."""
    RETRY_SECONDS=900
    MAX_SCAN=8
    MAX_EVENTS=4
    MAX_PAYLOAD_BYTES=2*MAX_CHUNK_BYTES

    def __init__(self,*args,anchor_reader,**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode!='log' or self.producer_id!='public-jsonl-v1':
            self.close();raise ValueError('retained retry is limited to SDK public logs')
        if not callable(anchor_reader):
            self.close();raise ValueError('original anchor reader required')
        self.anchor_reader=anchor_reader

    def retry_error(self):
        # An error remains visible until that unverified file retries successfully
        # or a validated receipt makes it ineligible. Never expose file identity.
        row=self.db.execute("SELECT m.value FROM source_files f JOIN file_sender_meta m ON m.key=('source-retry-error:'||f.producer||':'||f.file_id) WHERE f.producer=? AND f.verified=0 ORDER BY f.file_id LIMIT 1",(self.producer_id,)).fetchone()
        return json.loads(row[0])['code'] if row else None

    def send_once(self,limit=256):
        # Prioritize unsent originals: retry cannot consume their bounded budget
        # or advance a retry sweep while their first delivery is still pending.
        if not self.db.execute('SELECT 1 FROM source_pending LIMIT 1').fetchone():
            self.retry_once()
        return super().send_once(limit=limit)

    def retry_once(self,*,now=None):
        now=time.time() if now is None else now
        if not isinstance(now,(int,float)) or not 0<=now<float('inf'):raise ValueError('valid retry clock required')
        scan_key='source-retry-scan:'+self.producer_id
        self.db.execute('BEGIN IMMEDIATE')
        data=None;file_id=None
        try:
            found=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(scan_key,)).fetchone()
            cursor=found[0] if found else ''
            query="SELECT file_id,path,device,inode,business_start,offset,prefix_sha,manifest,capability,seal_event_id FROM source_files WHERE producer=? AND verified=0 AND manifest IS NOT NULL AND file_id>? ORDER BY file_id LIMIT ?"
            rows=self.db.execute(query,(self.producer_id,cursor,self.MAX_SCAN)).fetchall()
            if not rows and cursor:rows=self.db.execute(query,(self.producer_id,'',self.MAX_SCAN)).fetchall()
            advanced=cursor
            for file_id,path,dev,ino,start,offset,prefix,encoded,cap,seal_id in rows:
                advanced=file_id
                state_key='source-retry:'+self.producer_id+':'+file_id
                error_key='source-retry-error:'+self.producer_id+':'+file_id
                error_row=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(error_key,)).fetchone()
                if error_row and now<json.loads(error_row[0])['next_at']:continue
                old=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(state_key,)).fetchone()
                state=json.loads(old[0]) if old else {'index':0,'next_at':0}
                if (not isinstance(state,dict) or set(state)!={'index','next_at'} or type(state['index']) is not int
                        or state['index']<0 or type(state['next_at']) not in (int,float) or not math.isfinite(state['next_at']) or state['next_at']<0):raise ValueError('invalid retry state')
                info,data=self._read(path)
                if now-info.st_mtime<self.RETRY_SECONDS or now<state['next_at']:
                    data.close();data=None;continue
                manifest=validate_source_manifest(json.loads(encoded))
                if (manifest['producer_id']!=self.producer_id or manifest['generation']!=self.generation
                        or manifest['file_id']!=file_id or manifest['revision']!=0
                        or (info.st_dev,info.st_ino)!=(original_identity(dev),original_identity(ino)) or len(data)!=manifest['length']
                        or offset!=manifest['length'] or prefix!=manifest['sha256'] or data.sha()!=manifest['sha256']):
                    raise ValueError('retained source identity or bytes changed')
                anchor_row=self.db.execute('SELECT body FROM source_time_anchors WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
                if anchor_row is None:raise ValueError('source anchor unavailable')
                anchor=validate_native_time_anchor(json.loads(anchor_row[0]))
                if canonical(validate_native_time_anchor(self.anchor_reader(Path(path))))!=canonical(anchor):raise ValueError('source anchor changed')
                chunks=manifest['chunks']
                recorded=self.db.execute('SELECT offset,length,sha256,record_id FROM source_chunks WHERE producer=? AND file_id=? ORDER BY offset',(self.producer_id,file_id)).fetchall()
                if recorded!=[(c['offset'],c['length'],c['sha256'],c['record_id']) for c in chunks]:raise ValueError('source chunk manifest differs')
                if type(start) is not int or not 0<=start<=len(data) or not isinstance(cap,bytes) or len(cap)!=32:raise ValueError('source retry metadata unavailable')
                tail=data.rfind(b'\n')+1
                index=state['index'];count=0;budget=0
                if index>len(chunks):raise ValueError('retry cursor exceeds manifest')
                while index<=len(chunks) and count<self.MAX_EVENTS:
                    if index<len(chunks):
                        c=chunks[index];end=c['offset']+c['length']
                        if budget+c['length']>self.MAX_PAYLOAD_BYTES:break
                        payload=data[c['offset']:end]
                        if len(payload)!=c['length'] or hashlib.sha256(payload).hexdigest()!=c['sha256']:raise ValueError('retained chunk changed')
                        expected=f"{self.generation}:{file_id}:{c['offset']}:{c['sha256']}"
                        if c['record_id']!=expected:raise ValueError('source chunk identity differs')
                        if c['offset']<start<end or c['offset']<tail<end:raise ValueError('source chunk crosses purpose boundary')
                        claim=dict(producer_id=self.producer_id,generation=self.generation,file_id=file_id,revision=0,
                            offset=c['offset'],length=c['length'],sha256=c['sha256'],record_id=c['record_id'],
                            purpose='archive_only' if c['offset']<start or c['offset']>=tail else 'business')
                        event=SourceEnvelope.for_chunk(*self.identity,claim,payload);budget+=c['length']
                    else:
                        event=SourceEnvelope.for_seal(*self.identity,manifest,hashlib.sha256(cap).hexdigest(),native_time_anchor=anchor)
                        if event.event_id!=seal_id:raise ValueError('source seal identity differs')
                    pending=self.db.execute('SELECT producer,file_id,body FROM source_pending WHERE event_id=?',(event.event_id,)).fetchone()
                    if pending:
                        if pending!=(self.producer_id,file_id,event.to_json()):raise ValueError('pending replay identity conflict')
                    else:self._enqueue(file_id,event)
                    index+=1;count+=1
                data.check();data.close();data=None
                state={'index':0,'next_at':now+self.RETRY_SECONDS} if index>len(chunks) else {'index':index,'next_at':0}
                self.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)',(state_key,canonical(state)))
                self.db.execute('DELETE FROM file_sender_meta WHERE key=?',(error_key,))
                break  # at most one complete file or bounded part per invocation
            if rows:self.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)',(scan_key,advanced))
            self.db.commit()
        except Exception as error:
            self.db.rollback()
            # A failed file must not poison all subsequent sends. Persist only a
            # fixed category and a separate fair scan checkpoint after rollback.
            # Shared SQLite failure cannot be safely hidden or checkpointed.
            if file_id is None or isinstance(error,sqlite3.Error):raise
            code=('RETRY_OUTBOX_CAPACITY' if isinstance(error,BufferError) else
                  'RETRY_SOURCE_UNAVAILABLE' if isinstance(error,OSError) else
                  'RETRY_SOURCE_INVALID' if isinstance(error,(ValueError,TypeError,KeyError)) else 'RETRY_INTERNAL_ERROR')
            self.db.execute('BEGIN IMMEDIATE')
            try:
                self.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)',(scan_key,file_id))
                self.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)',('source-retry-error:'+self.producer_id+':'+file_id,canonical({'code':code,'next_at':now+self.RETRY_SECONDS})))
                self.db.commit()
            except BaseException:
                self.db.rollback();raise
        except BaseException:
            self.db.rollback();raise
        finally:
            if data is not None:data.close()
