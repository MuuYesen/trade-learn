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
        if evidence.get('status') not in ('FILLED','CANCELED','EXPIRED','REJECTED','DENIED'):
            raise ValueError('native terminal evidence required')
        with self._mutex,self._connection() as db:
            cursor=db.execute('UPDATE intents SET terminal=? WHERE client_oid=?',
                              (json.dumps(evidence,sort_keys=True),oid))
            if cursor.rowcount!=1:raise ValueError('native terminal has no admitted intent')

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
