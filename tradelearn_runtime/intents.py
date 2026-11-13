"""Durable admission intent ledger, never an order/fill accounting engine.

ExecutionHost calls mutations on its serial I/O worker. No history is deleted;
an absent broker order cannot turn a previously admitted intent into permission
for a new submission. The owning process retains an exclusive file lock.
"""
import json
from pathlib import Path
import sqlite3
from threading import RLock
import os
from contextlib import contextmanager


class IntentJournal:
    def __init__(self,path,*,identity):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self._mutex=RLock();self._closed=False
        descriptor=os.open(str(self.path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
        self._lock=os.fdopen(descriptor,'r+b')
        try:
            if os.name=='nt':
                import msvcrt
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(),msvcrt.LK_NBLCK,1)
                if os.fstat(self._lock.fileno()).st_size==0:
                    self._lock.write(b'0');self._lock.flush()
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self._lock.close();raise RuntimeError('intent journal already owned') from None
        try:
            with self._connection() as db:
                db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                row=db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
                if row and row[0]!=identity:raise ValueError('intent identity mismatch')
                db.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)",(identity,))
                db.execute('CREATE TABLE IF NOT EXISTS intents(client_oid TEXT PRIMARY KEY,payload TEXT NOT NULL,terminal TEXT)')
                db.execute('CREATE TABLE IF NOT EXISTS corrections(client_oid TEXT NOT NULL,correction_id TEXT NOT NULL,payload TEXT NOT NULL,previous_terminal TEXT,PRIMARY KEY(client_oid,correction_id))')
        except BaseException:
            self.close();raise

    @contextmanager
    def _connection(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    def prepare(self,oid,payload):
        with self._mutex,self._connection() as db:
            if self._closed:raise RuntimeError('intent journal closed')
            if db.execute('SELECT 1 FROM intents WHERE client_oid=?',(oid,)).fetchone():
                raise ValueError('durable client order ID reused; reconcile original intent')
            db.execute('INSERT INTO intents(client_oid,payload) VALUES(?,?)',
                       (oid,json.dumps(payload,sort_keys=True,separators=(',',':'))))

    def terminal(self,oid,evidence):
        if evidence.get('status') not in ('FILLED','CANCELED','EXPIRED','REJECTED','DENIED','VOIDED'):
            raise ValueError('native terminal evidence required')
        with self._mutex,self._connection() as db:
            cursor=db.execute('UPDATE intents SET terminal=? WHERE client_oid=?',
                              (json.dumps(evidence,sort_keys=True),oid))
            if cursor.rowcount!=1:raise ValueError('native terminal has no admitted intent')

    def corrected(self,oid,evidence):
        """Reopen only the same admitted CID, retaining its prior terminal proof."""
        if (not isinstance(evidence,dict) or evidence.get('client_oid')!=oid or
                type(evidence.get('is_reopened')) is not bool or
                not all(isinstance(evidence.get(k),str) and 0<len(evidence[k])<=512
                        for k in ('correction_id','event_id','account_id','strategy_id','instrument_id'))):
            raise ValueError('scoped native correction evidence required')
        status=evidence.get('native_status')
        if status not in (('SUBMITTED','ACCEPTED','PARTIALLY_FILLED') if evidence['is_reopened'] else ('VOIDED',)):
            raise ValueError('native correction terminal/reopen evidence differs')
        correction_key=json.dumps([evidence.get('child_client_oid',oid),evidence['correction_id']],separators=(',',':'))
        payload=json.dumps(evidence,sort_keys=True,separators=(',',':'))
        with self._mutex,self._connection() as db:
            row=db.execute('SELECT terminal FROM intents WHERE client_oid=?',(oid,)).fetchone()
            if row is None:raise ValueError('native correction has no original admitted intent')
            prior=db.execute('SELECT payload FROM corrections WHERE client_oid=? AND correction_id=?',
                (oid,correction_key)).fetchone()
            if prior is not None:
                if prior[0]!=payload:raise ValueError('conflicting native correction replay')
                return
            db.execute('INSERT INTO corrections VALUES(?,?,?,?)',(oid,correction_key,payload,row[0]))
            terminal=None if evidence['is_reopened'] else json.dumps(dict(status='VOIDED',
                filled_qty=evidence['filled_qty'],native_event_id=evidence['event_id'],correction=evidence),sort_keys=True)
            db.execute('UPDATE intents SET terminal=? WHERE client_oid=?',(terminal,oid))

    def unfinished(self):
        with self._mutex,self._connection() as db:
            return {oid:json.loads(payload) for oid,payload in db.execute('SELECT client_oid,payload FROM intents WHERE terminal IS NULL')}

    def close(self):
        with self._mutex:
            if self._closed:return
            self._closed=True
            if os.name=='nt':
                import msvcrt
                self._lock.seek(0);msvcrt.locking(self._lock.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(),fcntl.LOCK_UN)
            self._lock.close()
