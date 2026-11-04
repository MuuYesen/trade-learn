"""Background SQS transport and proof-gated GC of sealed native segments."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from pathlib import Path
import re
import threading
import time

from .source_retry import RetryingSourceFileSender
from .recovery import RecoveryIndex
from .publications import source_publication, retire_source_sidecars

from .archive_receipt_client import ArchiveReceiptClient
from .source_file_contract import validate_source_receipt, validate_native_time_anchor



class SegmentSender:
    """Native durable batches, one shared v2 writer; old cursor is migration input."""
    def __init__(self,settings,sqs,queue_url,*,receipt_client=None):
        if not settings.receipt_origin or not settings.archive_binding_id:
            raise ValueError('trusted HTTPS receipt origin required for source sender')
        self.settings=settings
        self._sidecar_error=False
        self.root=Path(settings.state_dir)
        folder=self.root/'sender';folder.mkdir(parents=True,exist_ok=True)
        old_path=folder/'outbox.sqlite'
        if old_path.exists():
            if old_path.is_symlink():raise ValueError('unsafe sender database')
            legacy=sqlite3.connect(old_path.as_uri()+'?mode=ro',uri=True)
            try:
                if legacy.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
                    row=legacy.execute("SELECT value FROM metadata WHERE key='connector_binding'").fetchone()
                    if row and json.loads(row[0])!=[settings.source,settings.strategy_id,settings.account_id,queue_url]:
                        raise ValueError('legacy sender binding changed')
            finally:legacy.close()
        self.receipts=receipt_client or ArchiveReceiptClient(settings.receipt_origin)
        self.recovery=RecoveryIndex(settings)
        self.outbox=RetryingSourceFileSender(folder/'outbox.sqlite',sqs,queue_url,settings.source,settings.strategy_id,settings.account_id,
                                     'public-jsonl-v1',archive_binding_id=settings.archive_binding_id,mode='log',max_retained_bytes=settings.max_disk_bytes,anchor_reader=self._native_anchor)

    def _migration_start(self,path):
        db=self.outbox.db
        if db.execute('SELECT 1 FROM source_files WHERE producer=? AND file_id=?',('public-jsonl-v1',path.stem)).fetchone():return None
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cursors'").fetchone():return 0
        row=db.execute('SELECT file_id,offset FROM cursors WHERE path=?',(str(path.resolve()),)).fetchone()
        fingerprint=None
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sealed'").fetchone():
            fingerprint=db.execute('SELECT digest,size FROM sealed WHERE segment=?',(path.name,)).fetchone()
        if row is not None and fingerprint is None:
            raise ValueError('legacy cursor lacks immutable sealed fingerprint; explicit evidence migration required')
        if fingerprint is not None:
            digest=hashlib.sha256();size=0
            with path.open('rb') as stream:
                while block:=stream.read(65536):digest.update(block);size+=len(block)
            if (digest.hexdigest(),size)!=tuple(fingerprint):raise ValueError('legacy sealed source fingerprint changed')
        if row is None:return 0
        if row[0]!=path.stem or not 0<=row[1]<=path.stat().st_size:
            raise ValueError('legacy segment cursor identity changed')
        return row[1]

    def _native_anchor(self,path):
        metadata=path.with_suffix('.native-time.json')
        try:
            from .filesystem import open_read_fd
            fd=open_read_fd(metadata)
            with os.fdopen(fd,'rb') as stream:
                info=os.fstat(stream.fileno())
                if os.name=='nt':
                    from .filesystem import verify_private_fd
                    verify_private_fd(stream.fileno())
                if metadata.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or (os.name!='nt' and stat.S_IMODE(info.st_mode)!=0o600) or info.st_size>1024:
                    raise ValueError('private bounded native anchor required')
                raw=stream.read(1025)
            return validate_native_time_anchor(json.loads(raw))
        except (OSError,ValueError) as exc:raise ValueError('explicit producer native anchor required; historical migration unresolved') from exc

    def _retire_source_metadata(self):
        result=retire_source_sidecars(self.outbox,self.root/'segments')
        self._sidecar_error=result['error']
        return result

    def maintenance_error(self):
        return 'source_sidecar_gc_failed' if self._sidecar_error else self.outbox.retry_error()

    def _capture_path(self,path):
        anchor=self._native_anchor(path)
        publication=source_publication(path,anchor)
        self.outbox.capture(path,path.stem,sealed=True if publication is None else publication['sealed'],
                            business_start=self._migration_start(path),native_time_anchor=anchor,
                            publication=publication)

    def send_once(self):
        # Existing proven segments can be retired before admitting new ones.
        # A receipt outage must not stop already durable messages being sent.
        receipt_error = None
        try:
            self.outbox.poll_receipts(self.receipts,before_collect=self.recovery.preserve)
        except Exception as exc:
            receipt_error = exc
        self._retire_source_metadata()
        sent=self.outbox.send_once()
        for path in sorted((self.root/'segments').glob('*.log')):
            if path.is_symlink() or not re.fullmatch(r'[a-f0-9]{32}-[0-9]{20}\.log',path.name):
                raise ValueError('unexpected sealed segment identity')
            self._capture_path(path)
        sent+=self.outbox.send_once()
        self.outbox.poll_receipts(self.receipts,before_collect=self.recovery.preserve)
        if receipt_error is not None:
            raise receipt_error
        return sent

    def close(self):self.outbox.close()


def require_sqs_dependencies():
    """Validate the optional SDK without creating a client or reading credentials."""
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:
        if exc.name not in {'boto3', 'botocore', 'botocore.config'}:
            raise
        raise ImportError("SQS sending requires 'trade-learn[sqs]'; install the matching local wheel with this extra") from exc
    return boto3, Config


def client_from_config(path):
    """Explicit send-only credentials; no default AWS chain or S3 client."""
    document = Path(path)
    if os.name=='nt':
        from .filesystem import check_private_path
        check_private_path(document)
    if os.name!='nt' and document.stat().st_mode & 0o077:
        raise ValueError('sender credentials file must be private (0600)')
    value = json.loads(document.read_text())
    if set(value) != {'region', 'queue_url', 'credentials_file'}:
        raise ValueError('invalid sender configuration')
    key_file = Path(value['credentials_file'])
    if os.name=='nt':check_private_path(key_file)
    if not key_file.is_absolute() or (os.name!='nt' and key_file.stat().st_mode & 0o077):
        raise ValueError('private absolute sender credentials path required')
    keys = json.loads(key_file.read_text())
    if (not {'access_key_id', 'secret_access_key'} <= keys.keys()
            or keys.keys() - {'access_key_id', 'secret_access_key', 'session_token'}
            or any(not isinstance(v, str) or not v for v in keys.values())):
        raise ValueError('invalid send-only credential object')
    boto3, Config = require_sqs_dependencies()
    sqs = boto3.client('sqs', region_name=value['region'], aws_access_key_id=keys['access_key_id'],
                       aws_secret_access_key=keys['secret_access_key'], aws_session_token=keys.get('session_token'),
                       config=Config(connect_timeout=5, read_timeout=10, retries={'max_attempts': 1}))
    return sqs, value['queue_url']


class BackgroundSender:
    def __init__(self, settings, client_factory=None, queue_url=None, *, interval=1, flush=None):
        self.settings, self.factory, self.queue_url = settings, client_factory, queue_url
        self.interval, self.flush = interval, flush
        self.stop = threading.Event()
        self.thread = None
        self._state = {'running': False, 'error': None, 'sent': 0, 'checked_at': None}

    def start(self):
        if self.thread is not None:
            raise RuntimeError('sender already started')
        self.thread = threading.Thread(target=self._run, name='leafquant-sqs-sender', daemon=True)
        self.thread.start()

    def _run(self):
        sender = None
        try:
            while not self.stop.is_set():
                try:
                    if sender is None:
                        if self.factory is None:
                            sqs, url = client_from_config(self.settings.sender_config)
                        else:
                            sqs, url = self.factory(), self.queue_url
                            if self.stop.is_set():
                                break
                        sender = SegmentSender(self.settings, sqs, url)
                    if self.factory is not None and self.stop.is_set():
                        break
                    flush_error = None
                    if self.flush is not None:
                        try:
                            self.flush()  # Native barrier releases GIL; callback never waits for SQS.
                        except Exception as exc:
                            # Sealed files remain independently eligible for proof-gated GC.
                            # Preserve the recorder failure in health even if delivery succeeds.
                            flush_error = type(exc).__name__
                    if self.factory is not None and self.stop.is_set():
                        break
                    count = sender.send_once()
                    self._state.update(running=True, error=flush_error or sender.maintenance_error(), sent=self._state['sent'] + count, checked_at=time.time())
                except Exception as exc:
                    self._state.update(running=True, error=type(exc).__name__, checked_at=time.time())
                    if self.factory is not None and sender is not None:
                        sender.close()
                        sender = None
                self.stop.wait(self.interval)
        finally:
            if sender is not None:
                sender.close()
            self._state['running'] = False

    def health(self):
        return dict(self._state)

    def close(self, timeout=30):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise TimeoutError('SQS sender is still draining; raw segments retained')
