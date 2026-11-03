"""Durable exact-byte live file transport. ACK is never an archive/GC proof.

A producer must fsync its published prefix before capture; immutable file_id is
never reused. SQLite transactions serialize multiple independent producers.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import secrets
import sqlite3
import stat
import time
import uuid
from threading import RLock

from .source_file_contract import MAX_CHUNK_BYTES, MAX_CHUNKS, MAX_FILE_BYTES, validate_source_manifest, validate_source_receipt, validate_native_time_anchor
from .sqs_transport import SourceEnvelope
from .filesystem import database_identity, original_identity, file_stat, path_stat, open_read_fd


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


class _FileBytes:
    """Pinned descriptor with at most one 256KiB read allocation."""
    def __init__(self,fd): self.fd=fd; self.info=file_stat(fd); self.length=self.info.st_size; self.published_digest=None;self.sealed=False;self._read_lock=RLock()
    def pin(self,length,*,sealed=False):
        if type(length) is not int or not 0<=length<=self.info.st_size:raise ValueError("invalid published source prefix")
        self.length=length;self.sealed=sealed;self.published_digest=self.sha()
    def __len__(self): return self.length
    def close(self): os.close(self.fd)
    def __getitem__(self,key):
        if not isinstance(key,slice): raise TypeError('bounded slice required')
        start=key.start or 0; end=self.length if key.stop is None else key.stop
        if not 0<=start<=end<=self.length or end-start>MAX_CHUNK_BYTES: raise ValueError('unbounded source read')
        with self._read_lock:
            os.lseek(self.fd,start,os.SEEK_SET)
            return os.read(self.fd,end-start)
    def rfind(self,needle,start=0,end=None):
        end=len(self) if end is None else end
        while end>start:
            left=max(start,end-MAX_CHUNK_BYTES)
            found=self[left:end].rfind(needle)
            if found>=0:return left+found
            end=left
        return -1
    def sha(self,length=None):
        length=len(self) if length is None else length
        h=hashlib.sha256()
        for offset in range(0,length,MAX_CHUNK_BYTES): h.update(self[offset:min(length,offset+MAX_CHUNK_BYTES)])
        return h.hexdigest()
    def check(self):
        now=file_stat(self.fd)
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if self.sealed and any(getattr(now,k)!=getattr(self.info,k) for k in fields):
            raise ValueError('sealed source changed during capture')
        if self.published_digest is not None:
            if (now.st_dev,now.st_ino)!=(self.info.st_dev,self.info.st_ino) or now.st_size<self.length or self.sha()!=self.published_digest:
                raise ValueError('published source prefix changed')
            return
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(now,k)!=getattr(self.info,k) for k in fields): raise ValueError('source changed during capture')


class SourceFileSender:
    def __init__(self, database, sqs, queue_url, source, strategy, account_id, producer_id, *, archive_binding_id, mode="log", max_pending_bytes=64*1024*1024, max_retained_bytes=256*1024*1024):
        if not isinstance(archive_binding_id,str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}",archive_binding_id) is None:raise ValueError("trusted archive binding required")
        self.archive_binding_id=archive_binding_id
        if mode not in ("log","binary"): raise ValueError("explicit source chunking mode required")
        self.mode,self.max_retained_bytes=mode,max_retained_bytes
        self.sqs,self.queue_url=sqs,queue_url
        self.identity=(source,strategy,account_id)
        self.producer_id=producer_id
        self.max_pending_bytes=max_pending_bytes
        database=Path(database)
        if os.name=='nt':
            from .filesystem import check_private_directory
            check_private_directory(database.parent)
        if database.is_symlink(): raise ValueError('unsafe sender database')
        self.db=sqlite3.connect(database,timeout=10,isolation_level=None)
        os.chmod(database,0o600)
        if os.name=='nt':
            from .filesystem import check_private_path
            check_private_path(database)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS file_sender_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS source_files(producer TEXT NOT NULL,file_id TEXT NOT NULL,path TEXT NOT NULL,
            device INTEGER NOT NULL,inode INTEGER NOT NULL,business_start INTEGER NOT NULL,offset INTEGER NOT NULL,
            prefix_sha TEXT NOT NULL,manifest TEXT,capability BLOB,seal_event_id TEXT,verified INTEGER NOT NULL DEFAULT 0,receipt TEXT,
            PRIMARY KEY(producer,file_id));
          CREATE TABLE IF NOT EXISTS source_chunks(producer TEXT NOT NULL,file_id TEXT NOT NULL,offset INTEGER NOT NULL,
            length INTEGER NOT NULL,sha256 TEXT NOT NULL,record_id TEXT NOT NULL,PRIMARY KEY(producer,file_id,offset));
          CREATE TABLE IF NOT EXISTS source_time_anchors(producer TEXT NOT NULL,file_id TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(producer,file_id));
          CREATE TABLE IF NOT EXISTS source_publications(producer TEXT NOT NULL,file_id TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(producer,file_id));
          CREATE TABLE IF NOT EXISTS source_diagnostics(producer TEXT NOT NULL,file_id TEXT NOT NULL,code TEXT NOT NULL,PRIMARY KEY(producer,file_id));
          CREATE TABLE IF NOT EXISTS receipt_retries(producer TEXT NOT NULL,file_id TEXT NOT NULL,next_attempt REAL NOT NULL,attempts INTEGER NOT NULL,PRIMARY KEY(producer,file_id));
          CREATE TABLE IF NOT EXISTS source_pending(ordinal INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,
            producer TEXT NOT NULL,file_id TEXT NOT NULL,body TEXT NOT NULL);
        ''')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('CREATE INDEX IF NOT EXISTS source_pending_producer_file ON source_pending(producer,file_id)')
            if 'receipt' not in [row[1] for row in self.db.execute('PRAGMA table_info(source_files)')]:
                self.db.execute('ALTER TABLE source_files ADD COLUMN receipt TEXT')
            columns=[row[1] for row in self.db.execute('PRAGMA table_info(receipt_retries)')]
            if columns==['file_id','next_attempt','attempts']:
                self.db.execute('ALTER TABLE receipt_retries RENAME TO receipt_retries_migrating')
                self.db.execute('CREATE TABLE receipt_retries(producer TEXT NOT NULL,file_id TEXT NOT NULL,next_attempt REAL NOT NULL,attempts INTEGER NOT NULL,PRIMARY KEY(producer,file_id))')
                self.db.execute('INSERT INTO receipt_retries SELECT f.producer,r.file_id,r.next_attempt,r.attempts FROM receipt_retries_migrating r JOIN source_files f ON f.file_id=r.file_id')
                self.db.execute('DROP TABLE receipt_retries_migrating')
            binding=canonical([source,strategy,account_id,queue_url,archive_binding_id])
            self.db.execute("INSERT OR IGNORE INTO file_sender_meta VALUES('binding',?)",(binding,))
            if self.db.execute("SELECT value FROM file_sender_meta WHERE key='binding'").fetchone()[0]!=binding:
                raise ValueError('source sender binding changed')
            self.db.execute("INSERT OR IGNORE INTO file_sender_meta VALUES('generation',?)",(uuid.uuid4().hex,))
            self.generation=self.db.execute("SELECT value FROM file_sender_meta WHERE key='generation'").fetchone()[0]
            mode_key='mode:'+producer_id
            self.db.execute('INSERT OR IGNORE INTO file_sender_meta VALUES(?,?)',(mode_key,mode))
            if self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(mode_key,)).fetchone()[0]!=mode:
                raise ValueError('producer source chunking mode changed')
            self.db.commit()
        except BaseException:
            self.db.rollback(); self.db.close(); raise

    def close(self): self.db.close()

    def _read(self,path):
        if os.name=='nt':
            from .filesystem import check_private_directory
            check_private_directory(Path(path).parent)
        before=path_stat(path)
        if not stat.S_ISREG(before.st_mode) or getattr(before,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0):
            raise ValueError('unsafe source leaf')
        fd=open_read_fd(path)
        data=_FileBytes(fd)
        if os.name=='nt':
            from .filesystem import verify_private_fd
            try:verify_private_fd(fd)
            except BaseException:data.close();raise
        # Windows has no O_NOFOLLOW; comparing the opened descriptor to the
        # lstat identity prevents a raced symlink/reparse replacement being read.
        if any(getattr(data.info,key)!=getattr(before,key) for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):
            data.close();raise ValueError('source identity changed during open')
        info=data.info
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>MAX_FILE_BYTES:
            data.close(); raise ValueError('unsafe source file')
        return info,data

    def capture(self,path,file_id,*,native_time_anchor,sealed=False,business_start=None,publication=None):
        """Capture raw byte pieces up to the live LF prefix; sealed tails are archive_only.

        business_start is an explicit one-shot old ACK watermark. Prefix bytes
        are archive_only; it is pinned permanently and cannot rewind counters.
        """
        anchor=canonical(validate_native_time_anchor(native_time_anchor))
        path=Path(path).absolute()
        try: info,data=self._read(path)
        except OSError as exc: raise ValueError('source file unavailable') from exc
        self.db.execute('BEGIN IMMEDIATE')
        try:
            previous=self.db.execute('SELECT body FROM source_time_anchors WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
            if previous and previous[0]!=anchor:raise ValueError('native time anchor changed')
            if not previous and self.db.execute('SELECT 1 FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone():raise ValueError('historical source lacks explicit native anchor migration')
            self.db.execute('INSERT OR IGNORE INTO source_time_anchors VALUES(?,?,?)',(self.producer_id,file_id,anchor))
            if publication is not None:
                keys={'device','inode','published_length','publication_sequence','sealed','owner_generation'}
                if (not isinstance(publication,dict) or set(publication)!=keys
                        or any(type(publication[k]) is not int or publication[k]<0 for k in ('device','inode','published_length','publication_sequence'))
                        or type(publication['sealed']) is not bool or publication['sealed']!=sealed
                        or not isinstance(publication['owner_generation'],str) or not publication['owner_generation']
                        or (publication['device'],publication['inode'])!=(info.st_dev,info.st_ino)
                        or (sealed and publication['published_length']!=info.st_size)):
                    raise ValueError('invalid trusted source publication')
                encoded=canonical(publication)
                prior=self.db.execute('SELECT body FROM source_publications WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
                if prior:
                    old=json.loads(prior[0])
                    if (old['owner_generation']!=publication['owner_generation']
                            or old['publication_sequence']>publication['publication_sequence']
                            or old['published_length']>publication['published_length']
                            or ((old['sealed'] or old['publication_sequence']==publication['publication_sequence']) and prior[0]!=encoded)):
                        raise ValueError('native publication regressed or changed')
                data.pin(publication['published_length'],sealed=sealed)
                self.db.execute('INSERT OR REPLACE INTO source_publications VALUES(?,?,?)',(self.producer_id,file_id,encoded))
            row=self.db.execute('SELECT path,device,inode,business_start,offset,prefix_sha,manifest FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
            if row:
                oldpath,dev,ino,start,offset,prefix,manifest=row
                if (str(path),info.st_dev,info.st_ino)!=(oldpath,original_identity(dev),original_identity(ino)) or len(data)<offset or data.sha(offset)!=prefix:
                    raise ValueError('source identity or acknowledged prefix changed')
                if business_start is not None and business_start!=start: raise ValueError('business watermark changed')
                if manifest:
                    if data.sha()!=json.loads(manifest)['sha256'] or len(data)!=json.loads(manifest)['length']:
                        raise ValueError('sealed source changed')
                    data.check(); self.db.commit(); return
            else:
                start=0 if business_start is None else business_start
                if type(start) is not int or not 0<=start<=len(data) or (self.mode=='log' and start and data[start-1:start]!=b'\n'):
                    raise ValueError('invalid business watermark')
                offset=0
                self.db.execute('INSERT INTO source_files(producer,file_id,path,device,inode,business_start,offset,prefix_sha) VALUES(?,?,?,?,?,?,?,?)',(self.producer_id,file_id,str(path),database_identity(info.st_dev),database_identity(info.st_ino),start,0,hashlib.sha256(b'').hexdigest()))
            tail_start=data.rfind(b'\n')+1 if self.mode=='log' else len(data)
            if sealed and tail_start<len(data):
                self.db.execute('INSERT OR REPLACE INTO source_diagnostics VALUES(?,?,?)',(self.producer_id,file_id,'unterminated_tail'))
            retained=0
            for item in self.db.execute('SELECT path,verified FROM source_files WHERE verified!=2'):
                try:retained+=os.lstat(item[0]).st_size
                except FileNotFoundError:
                    if item[1]!=1:raise ValueError('unverified source disappeared')
            if retained>self.max_retained_bytes: raise BufferError('unproven source retention budget exhausted')
            end=len(data) if sealed or self.mode=='binary' else data.rfind(b'\n')+1
            cursor=offset
            count=self.db.execute('SELECT count(*) FROM source_chunks WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()[0]
            while cursor<end:
                limit=min(end,cursor+MAX_CHUNK_BYTES)
                if cursor<start<limit: limit=start
                if cursor<tail_start<limit: limit=tail_start
                if limit<=cursor: raise BufferError('source line exceeds chunk budget')
                payload=data[cursor:limit]
                digest=hashlib.sha256(payload).hexdigest()
                record_id=f'{self.generation}:{file_id}:{cursor}:{digest}'
                claim=dict(producer_id=self.producer_id,generation=self.generation,file_id=file_id,revision=0,offset=cursor,length=len(payload),sha256=digest,record_id=record_id,purpose='archive_only' if cursor<start or (self.mode=='log' and cursor>=tail_start) else 'business')
                envelope=SourceEnvelope.for_chunk(*self.identity,claim,payload)
                self._enqueue(file_id,envelope)
                count+=1
                if count>MAX_CHUNKS: raise BufferError('source chunk budget requires rotation')
                self.db.execute('INSERT INTO source_chunks VALUES(?,?,?,?,?,?)',(self.producer_id,file_id,cursor,len(payload),digest,record_id))
                cursor=limit
            self.db.execute('UPDATE source_files SET offset=?,prefix_sha=? WHERE producer=? AND file_id=?',(end,data.sha(end),self.producer_id,file_id))
            if sealed:
                chunks=[dict(offset=r[0],length=r[1],sha256=r[2],record_id=r[3]) for r in self.db.execute('SELECT offset,length,sha256,record_id FROM source_chunks WHERE producer=? AND file_id=? ORDER BY offset',(self.producer_id,file_id))]
                manifest=validate_source_manifest(dict(schema_version=1,producer_id=self.producer_id,generation=self.generation,file_id=file_id,revision=0,length=len(data),sha256=data.sha(),chunks=chunks))
                cap=secrets.token_bytes(32)
                envelope=SourceEnvelope.for_seal(*self.identity,manifest,hashlib.sha256(cap).hexdigest(),native_time_anchor=json.loads(anchor))
                self._enqueue(file_id,envelope)
                self.db.execute('UPDATE source_files SET manifest=?,capability=?,seal_event_id=? WHERE producer=? AND file_id=?',(canonical(manifest),cap,envelope.event_id,self.producer_id,file_id))
            data.check(); self.db.commit()
        except BaseException:
            self.db.rollback(); raise
        finally:
            data.close()

    def _enqueue(self,file_id,envelope):
        body=envelope.to_json()
        size=self.db.execute('SELECT coalesce(sum(length(CAST(body AS BLOB))),0) FROM source_pending').fetchone()[0]
        if size+len(body.encode())>self.max_pending_bytes: raise BufferError('source durable outbox full')
        self.db.execute('INSERT INTO source_pending(event_id,producer,file_id,body) VALUES(?,?,?,?)',(envelope.event_id,self.producer_id,file_id,body))

    def send_once(self,limit=256):
        sent=0
        for _ in range(min(limit,256)):
            row=self.db.execute('SELECT ordinal,event_id,body FROM source_pending ORDER BY ordinal LIMIT 1').fetchone()
            if row is None: break
            ordinal,event_id,body=row
            envelope=SourceEnvelope.from_json(body)
            if envelope.event_id!=event_id:raise ValueError('pending source event identity differs')
            self.sqs.send_message(QueueUrl=self.queue_url,MessageBody=body,MessageGroupId=f'{self.identity[1]}:{self.identity[2]}',MessageDeduplicationId=event_id)
            self.db.execute('DELETE FROM source_pending WHERE ordinal=? AND event_id=?',(ordinal,event_id)); sent+=1
        return sent

    def _prune_retired_once(self):
        # Scan bounded primary-key ranges per producer, cycling after EOF. A
        # skipped pending/active row is reconsidered on a subsequent cycle.
        budget=64
        self.db.execute('BEGIN IMMEDIATE')
        try:
            chunk_key='retired_gc_chunks:'+self.producer_id
            retry_key='retired_gc_retries:'+self.producer_id
            chunk_row=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(chunk_key,)).fetchone()
            retry_row=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(retry_key,)).fetchone()
            chunk_cursor=json.loads(chunk_row[0]) if chunk_row else None
            retry_cursor=json.loads(retry_row[0]) if retry_row else None
            args=[self.producer_id]
            chunk_where='producer=?'
            if chunk_cursor is not None:
                chunk_where+=' AND (file_id,offset)>(?,?)'
                args.extend(chunk_cursor)
            chunks=self.db.execute('SELECT file_id,offset FROM source_chunks WHERE '+chunk_where+' ORDER BY file_id,offset LIMIT ?',(*args,budget)).fetchall()
            args=[self.producer_id]
            retry_where='producer=?'
            if retry_cursor is not None:
                retry_where+=' AND file_id>?'
                args.append(retry_cursor)
            retries=self.db.execute('SELECT file_id FROM receipt_retries WHERE '+retry_where+' ORDER BY file_id LIMIT ?',(*args,budget)).fetchall()
            eligible=' AND EXISTS (SELECT 1 FROM source_files f WHERE f.producer=? AND f.file_id=? AND f.verified=2) AND NOT EXISTS (SELECT 1 FROM source_pending p WHERE p.producer=? AND p.file_id=?)'
            for file_id,offset in chunks:
                self.db.execute('DELETE FROM source_chunks WHERE producer=? AND file_id=? AND offset=?'+eligible,(self.producer_id,file_id,offset,self.producer_id,file_id,self.producer_id,file_id))
            for (file_id,) in retries:
                self.db.execute('DELETE FROM receipt_retries WHERE producer=? AND file_id=?'+eligible,(self.producer_id,file_id,self.producer_id,file_id,self.producer_id,file_id))
            for key,value in ((chunk_key,chunks[-1] if len(chunks)==budget else None),(retry_key,retries[-1][0] if len(retries)==budget else None)):
                if value is None:self.db.execute('DELETE FROM file_sender_meta WHERE key=?',(key,))
                else:self.db.execute('INSERT OR REPLACE INTO file_sender_meta(key,value) VALUES(?,?)',(key,json.dumps(value)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def poll_receipts(self,client,*,before_collect=None,limit=16,now=None,prefer_recent=False):
        self._retry_windows_gc()
        now=time.time() if now is None else now
        if type(prefer_recent) is not bool:raise ValueError('explicit receipt scheduling mode required')
        self._prune_retired_once()
        if prefer_recent:
            if type(limit) is not int or not 1<=limit<=16:raise ValueError('bounded receipt poll required')
            query='SELECT file_id,path,manifest,verified FROM source_files WHERE producer=? AND verified!=2 AND manifest IS NOT NULL AND file_id NOT IN (SELECT file_id FROM receipt_retries WHERE producer=? AND next_attempt>?)'
            old=self.db.execute(query+' ORDER BY rowid ASC LIMIT ?', (self.producer_id,self.producer_id,now,limit)).fetchall()
            new=self.db.execute(query+' ORDER BY rowid DESC LIMIT ?', (self.producer_id,self.producer_id,now,limit)).fetchall()
            old_count=limit-limit//2
            rows=list(old[:old_count]);seen={row[0] for row in rows}
            for row in [*new,*old]:
                if len(rows)>=limit:break
                if row[0] not in seen:rows.append(row);seen.add(row[0])
        else:
            rows=self.db.execute('SELECT file_id,path,manifest,verified FROM source_files WHERE producer=? AND verified!=2 AND manifest IS NOT NULL AND file_id NOT IN (SELECT file_id FROM receipt_retries WHERE producer=? AND next_attempt>?) LIMIT ?',(self.producer_id,self.producer_id,now,min(limit,16))).fetchall()
        collected=0
        for file_id,path,raw_manifest,verified in rows:
            previous=self.db.execute('SELECT attempts FROM receipt_retries WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
            attempts=previous[0]+1 if previous else 1
            self.db.execute('INSERT OR REPLACE INTO receipt_retries VALUES(?,?,?,?)',(self.producer_id,file_id,now+min(300,5*2**min(attempts-1,6)),attempts))
            proof=client.fetch(**self.receipt_request(file_id))
            if proof is None:continue
            validate_source_receipt(proof,json.loads(raw_manifest),expected_binding_id=self.archive_binding_id)
            if before_collect is not None:
                if Path(path).exists():before_collect(path)
                elif verified!=1:raise ValueError('source disappeared before retention checkpoint')
            self.collect(file_id,proof);collected+=1
        return collected

    def receipt_request(self,file_id):
        row=self.db.execute('SELECT seal_event_id,capability FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
        if not row or row[0] is None: return None
        return dict(seal_event_id=row[0],token=base64.urlsafe_b64encode(row[1]).decode().rstrip('='))

    def verified_proof(self,file_id):
        meta=self.db.execute('SELECT verified,coalesce(length(CAST(receipt AS BLOB)),0),coalesce(length(CAST(manifest AS BLOB)),0) FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
        if meta is None or meta[0] not in (1,2):return None
        if not 0<meta[1]<=4096 or not 0<meta[2]<=256*1024:raise ValueError('invalid persisted complete proof budget')
        receipt,manifest=self.db.execute('SELECT receipt,manifest FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
        return validate_source_receipt(json.loads(receipt),json.loads(manifest),expected_binding_id=self.archive_binding_id)

    def collect(self,file_id,proof):
        row=self.db.execute('SELECT path,device,inode,manifest,verified FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
        if not row or row[3] is None: raise ValueError('source is not sealed')
        manifest=json.loads(row[3]); proof=validate_source_receipt(proof,manifest,expected_binding_id=self.archive_binding_id)
        path=Path(row[0])
        try: info,data=self._read(path)
        except FileNotFoundError:
            if row[4] not in (1,2): raise ValueError('unverified source disappeared')
            self._finish_gc(path,file_id); return
        try:
            if (info.st_dev,info.st_ino)!=(original_identity(row[1]),original_identity(row[2])) or len(data)!=manifest['length'] or data.sha()!=manifest['sha256']:
                raise ValueError('source changed before collection')
            data.check()
        finally:
            data.close()

        # Producers own this immutable sealed path and must never replace/reuse it.
        current=path_stat(path)
        if (current.st_dev,current.st_ino,current.st_mtime_ns,current.st_ctime_ns)!=(info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns): raise ValueError('source changed before unlink')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('UPDATE source_files SET verified=1,receipt=? WHERE producer=? AND file_id=?',(canonical(proof),self.producer_id,file_id)); self.db.commit()
        except BaseException: self.db.rollback(); raise
        self._finish_gc(path,file_id,True)

    def _finish_gc(self,path,file_id,unlink=False):
        if os.name=='nt':
            return self._finish_windows_gc(path,file_id,unlink)
        # This existing GC synchronization point must establish capability
        # BEFORE destructive IO. Unsupported platforms retain their raw file;
        # no new Windows deletion/flush implementation is implied here.
        if not hasattr(os,'O_DIRECTORY'):raise ValueError('durable directory sync unavailable; source retained')
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            if not stat.S_ISDIR(os.fstat(fd).st_mode):raise ValueError('source parent is not a directory')
            os.fsync(fd) # Fail closed before unlink on an unsupported filesystem.
            if unlink:path.unlink()
            os.fsync(fd)
        finally:os.close(fd)
        self.db.execute('UPDATE source_files SET verified=2 WHERE producer=? AND file_id=?',(self.producer_id,file_id))


    def _finish_windows_gc(self,path,file_id,unlink):
        from .filesystem import move_write_through, verify_private_fd
        proof=self.verified_proof(file_id)
        if proof is None:raise ValueError('complete persisted proof required before retirement')
        row=self.db.execute('SELECT device,inode,manifest,verified FROM source_files WHERE producer=? AND file_id=?',(self.producer_id,file_id)).fetchone()
        manifest=json.loads(row[2]);path=Path(path)
        retired=path.with_name('.'+path.name+'.proof-retired')
        def checked(candidate):
            info,data=self._read(candidate)
            try:
                verify_private_fd(data.fd)
                if (info.st_dev,info.st_ino)!=(original_identity(row[0]),original_identity(row[1])) or len(data)!=manifest['length'] or data.sha()!=manifest['sha256']:
                    raise ValueError('proof retirement source changed')
                data.check()
            finally:data.close()
        if os.path.lexists(path):
            if not unlink:raise ValueError('source reappeared during retirement')
            if os.path.lexists(retired):raise ValueError('retirement destination already exists')
            checked(path)
            move_write_through(path,retired)
        if os.path.lexists(retired):checked(retired)
        # verified=1 and the complete proof were committed before the rename.
        # A restart at either side of the rename can therefore finish safely.
        self.db.execute('UPDATE source_files SET verified=2 WHERE producer=? AND file_id=?',(self.producer_id,file_id))
        if os.path.lexists(retired):retired.unlink()
        # DeleteFile is reclamation only. If it is lost or fails, this same
        # proof-bound tombstone is retried; it is never discovered as a log.

    def _retry_windows_gc(self,limit=16):
        if os.name!='nt':return
        key='windows-proof-retirement:'+self.producer_id
        row=self.db.execute('SELECT value FROM file_sender_meta WHERE key=?',(key,)).fetchone();cursor=row[0] if row else ''
        query='SELECT file_id,path,verified FROM source_files WHERE producer=? AND verified IN (1,2) AND file_id>? ORDER BY file_id LIMIT ?'
        rows=self.db.execute(query,(self.producer_id,cursor,limit)).fetchall()
        if not rows and cursor:rows=self.db.execute(query,(self.producer_id,'',limit)).fetchall()
        error=None
        for file_id,path,verified in rows:
            original=Path(path);retired=original.with_name('.'+original.name+'.proof-retired')
            try:
                if verified==1 or not os.path.lexists(original) or os.path.lexists(retired):self._finish_windows_gc(original,file_id,verified==1 and os.path.lexists(original))
            except (OSError,ValueError) as exc:error=error or exc
        if rows:self.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)',(key,rows[-1][0]))
        if error is not None:raise error
