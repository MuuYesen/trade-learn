"""SQS FIFO transport for existing complete log lines and snapshot batches.

The wire envelope does not rename or reinterpret business fields. Callers must
validate source records before sending and processing them. boto3 is injected so
the shared monitoring core has no AWS dependency at import time.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import zlib


# SQS Standard/FIFO support 1 MiB since 2025-08-04; sender and receiver share this bound.
MAX_BODY_BYTES = 1024 * 1024
# A complete four-file batch can compress below the wire cap while its
# original bytes exceed it. Bound decoded snapshots separately on both ends.
MAX_SNAPSHOT_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_LOG_PAYLOAD_BYTES = 256 * 1024
MAX_LOG_READ_BYTES = 96 * 1024
KINDS = frozenset({"log", "snapshot", "heartbeat"})


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or any(ord(c) < 32 for c in value):
        raise ValueError("invalid transport identity")
    return value


@dataclass(frozen=True)
class Envelope:
    source: str
    strategy: str
    account_id: str
    kind: str
    record_id: str
    payload: bytes
    digest: str
    event_id: str

    @classmethod
    def for_bytes(cls, source: str, strategy: str, account_id: str, kind: str,
                  record_id: str, payload: bytes) -> "Envelope":
        for value in (source, strategy, account_id, record_id):
            _identifier(value)
        if kind not in KINDS or not isinstance(payload, bytes) or not payload:
            raise ValueError("invalid transport payload")
        limit = MAX_SNAPSHOT_PAYLOAD_BYTES if kind == "snapshot" else MAX_LOG_PAYLOAD_BYTES
        if len(payload) > limit:
            raise ValueError("transport payload exceeds decoded size limit")
        digest = hashlib.sha256(payload).hexdigest()
        token = json.dumps([source, strategy, account_id, kind, record_id, digest],
                           ensure_ascii=False, separators=(",", ":")).encode()
        event_id = hashlib.sha256(token).hexdigest()
        result = cls(source, strategy, account_id, kind, record_id, payload, digest, event_id)
        result.to_json()  # Reject oversize messages before persisting a cursor.
        return result

    def to_json(self) -> str:
        body = json.dumps({
            "version": 1,
            "source": self.source,
            "strategy": self.strategy,
            "account_id": self.account_id,
            "kind": self.kind,
            "record_id": self.record_id,
            "digest": self.digest,
            "event_id": self.event_id,
            "payload_zlib_base64": base64.b64encode(zlib.compress(self.payload)).decode("ascii"),
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(body.encode()) > MAX_BODY_BYTES:
            raise ValueError("SQS message exceeds configured transport limit")
        return body

    @classmethod
    def from_json(cls, raw: str) -> "Envelope":
        if not isinstance(raw, str) or len(raw.encode()) > MAX_BODY_BYTES:
            raise ValueError("invalid SQS message size")
        try:
            doc = json.loads(raw)
            if isinstance(doc, dict) and type(doc.get('version')) is int and doc['version'] == 2:
                return SourceEnvelope.from_json(raw)
            if set(doc) != {"version", "source", "strategy", "account_id", "kind",
                            "record_id", "digest", "event_id", "payload_zlib_base64"} or doc["version"] != 1:
                raise ValueError("unsupported transport envelope")
            if doc["kind"] not in KINDS:
                raise ValueError("unsupported transport kind")
            limit = MAX_SNAPSHOT_PAYLOAD_BYTES if doc["kind"] == "snapshot" else MAX_LOG_PAYLOAD_BYTES
            compressed = base64.b64decode(doc["payload_zlib_base64"], validate=True)
            inflater = zlib.decompressobj()
            payload = inflater.decompress(compressed, limit + 1)
            if not inflater.eof or inflater.unused_data or len(payload) > limit:
                raise ValueError("invalid compressed payload")
            expected = cls.for_bytes(doc["source"], doc["strategy"], doc["account_id"],
                                     doc["kind"], doc["record_id"], payload)
            if expected.digest != doc["digest"] or expected.event_id != doc["event_id"]:
                raise ValueError("transport checksum mismatch")
            return expected
        except (KeyError, TypeError, json.JSONDecodeError, zlib.error, ValueError) as exc:
            raise ValueError("invalid transport envelope") from exc


def _unique_source_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate source wire key')
        result[key] = value
    return result


@dataclass(frozen=True)
class SourceEnvelope(Envelope):
    """Single v2 source-file wire. Chunk bytes are never reserialized."""
    source_file: dict | None
    receipt_capability_sha256: str | None
    native_time_anchor: dict | None

    @property
    def version(self):
        return 2

    @classmethod
    def _build(cls, source, strategy, account_id, kind, record_id, payload, claim, capability, anchor):
        for value in (source, strategy, account_id, record_id):
            _identifier(value)
        digest = hashlib.sha256(payload).hexdigest()
        identity = [2, source, strategy, account_id, kind, record_id, digest, claim, capability]
        if kind == 'source_seal': identity.append(anchor)
        token = json.dumps(identity,
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        result = cls(source, strategy, account_id, kind, record_id, payload, digest,
                     hashlib.sha256(token).hexdigest(), claim, capability, anchor)
        result.to_json()
        return result

    @classmethod
    def for_chunk(cls, source, strategy, account_id, claim, payload):
        from .source_file_contract import validate_chunk_claim
        checked = validate_chunk_claim(claim, payload)
        return cls._build(source, strategy, account_id, 'source_chunk', checked['record_id'],
                          payload, checked, None, None)

    @classmethod
    def for_seal(cls, source, strategy, account_id, manifest, receipt_capability_sha256, *, native_time_anchor):
        from .source_file_contract import validate_source_manifest, source_manifest_digest, validate_native_time_anchor
        checked = validate_source_manifest(manifest)
        capability = receipt_capability_sha256
        if not isinstance(capability, str) or not re.fullmatch('[0-9a-f]{64}', capability):
            raise ValueError('exact receipt capability SHA256 required')
        payload = json.dumps(checked, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        return cls._build(source, strategy, account_id, 'source_seal',
                          'seal:' + source_manifest_digest(checked), payload, None, capability, validate_native_time_anchor(native_time_anchor))

    def to_json(self):
        body = json.dumps(dict(version=2, source=self.source, strategy=self.strategy,
            account_id=self.account_id, kind=self.kind, record_id=self.record_id,
            digest=self.digest, event_id=self.event_id,
            payload_zlib_base64=base64.b64encode(zlib.compress(self.payload)).decode('ascii'),
            source_file=self.source_file, receipt_capability_sha256=self.receipt_capability_sha256, native_time_anchor=self.native_time_anchor),
            sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if len(body.encode()) > MAX_BODY_BYTES:
            raise ValueError('source wire exceeds transport limit')
        return body

    @classmethod
    def from_json(cls, raw):
        from .source_file_contract import MAX_CHUNK_BYTES, MAX_MANIFEST_BYTES
        try:
            if not isinstance(raw, str) or len(raw.encode()) > MAX_BODY_BYTES:
                raise ValueError('invalid source wire size')
            doc = json.loads(raw, object_pairs_hook=_unique_source_object)
            keys = {'version','source','strategy','account_id','kind','record_id','digest','event_id',
                    'payload_zlib_base64','source_file','receipt_capability_sha256','native_time_anchor'}
            if not isinstance(doc, dict) or set(doc) != keys or type(doc['version']) is not int or doc['version'] != 2:
                raise ValueError('invalid source wire fields')
            if doc['kind'] not in {'source_chunk','source_seal'}:
                raise ValueError('invalid source wire kind')
            limit = MAX_CHUNK_BYTES if doc['kind'] == 'source_chunk' else MAX_MANIFEST_BYTES
            encoded = base64.b64decode(doc['payload_zlib_base64'], validate=True)
            decoder = zlib.decompressobj()
            payload = decoder.decompress(encoded, limit + 1)
            if not decoder.eof or decoder.unused_data or not payload or len(payload) > limit:
                raise ValueError('invalid source compressed payload')
            if doc['kind'] == 'source_chunk':
                if doc['receipt_capability_sha256'] is not None or doc['native_time_anchor'] is not None:
                    raise ValueError('chunk cannot carry seal capability')
                result = cls.for_chunk(doc['source'],doc['strategy'],doc['account_id'],doc['source_file'],payload)
            else:
                if doc['source_file'] is not None:
                    raise ValueError('seal cannot carry chunk claim')
                result = cls.for_seal(doc['source'],doc['strategy'],doc['account_id'],json.loads(payload),doc['receipt_capability_sha256'],native_time_anchor=doc['native_time_anchor'])
                if result.payload != payload:
                    raise ValueError('seal payload must be canonical manifest')
            if (result.record_id,result.digest,result.event_id) != (doc['record_id'],doc['digest'],doc['event_id']):
                raise ValueError('source wire identity or checksum mismatch')
            return result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, zlib.error) as exc:
            raise ValueError('invalid source envelope') from exc


class FileOutbox:
    """Send completed UTF-8 log lines; commit local cursor only after SQS ACK."""

    def __init__(self, state_path, sqs_client, queue_url, source, strategy, account_id, *, file_identity=None):
        self.file_identity = file_identity
        self.sqs, self.queue_url = sqs_client, _identifier(queue_url)
        self.source, self.strategy, self.account_id = (
            _identifier(source), _identifier(strategy), _identifier(account_id))
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(state_path))
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cursors (path TEXT PRIMARY KEY, file_id TEXT NOT NULL, offset INTEGER NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _file_id(self, path, stat):
        return (_identifier(self.file_identity(path, stat)) if self.file_identity is not None
                else f"{stat.st_dev}:{stat.st_ino}")

    def cursor(self, path) -> int:
        path = Path(path)
        stat = path.stat()
        file_id = self._file_id(path, stat)
        row = self.db.execute("SELECT file_id, offset FROM cursors WHERE path=?", (str(path.resolve()),)).fetchone()
        return row[1] if row and row[0] == file_id and row[1] <= stat.st_size else 0

    def bootstrap_existing(self, paths) -> bool:
        """One-time cutover boundary; future rotated files always start at zero."""
        if self.db.execute("SELECT 1 FROM metadata WHERE key='bootstrap_done'").fetchone():
            return False
        if self.db.execute("SELECT 1 FROM cursors LIMIT 1").fetchone():
            raise ValueError("cannot bootstrap an outbox with existing cursors")
        with self.db:
            for path in paths:
                path = Path(path)
                stat = path.stat()
                self.db.execute("INSERT INTO cursors VALUES (?, ?, ?)",
                                (str(path.resolve()), self._file_id(path, stat), stat.st_size))
            self.db.execute("INSERT INTO metadata VALUES ('bootstrap_done', 'tail')")
        return True

    def send_new_lines(self, path) -> int:
        path = Path(path)
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            file_id = self._file_id(path, stat)
            key = str(path.resolve())
            row = self.db.execute("SELECT file_id, offset FROM cursors WHERE path=?", (key,)).fetchone()
            offset = row[1] if row and row[0] == file_id and row[1] <= stat.st_size else 0
            stream.seek(offset)
            block = stream.read(MAX_LOG_READ_BYTES)
            end = block.rfind(b"\n")
            if end < 0:
                if len(block) == MAX_LOG_READ_BYTES:
                    raise ValueError("strategy log line exceeds transport limit")
                return 0
            payload = block[:end + 1]
            payload.decode("utf-8")
            record_id = f"{file_id}:{offset}:{offset + len(payload)}"
            envelope = Envelope.for_bytes(self.source, self.strategy, self.account_id,
                                          "log", record_id, payload)
            self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=envelope.to_json(),
                MessageGroupId=f"{self.strategy}:{self.account_id}",
                MessageDeduplicationId=envelope.event_id,
            )
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO cursors VALUES (?, ?, ?)",
                                (key, file_id, offset + len(payload)))
            return 1

    def close(self):
        self.db.close()


class SnapshotOutbox:
    """Send one already-validated four-file revision without changing its bytes."""

    def __init__(self, state_path, sqs_client, queue_url, source, strategy, account_id):
        self.sqs, self.queue_url = sqs_client, _identifier(queue_url)
        self.source, self.strategy, self.account_id = (
            _identifier(source), _identifier(strategy), _identifier(account_id))
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(state_path))
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS sent (event_id TEXT PRIMARY KEY)")
        self.db.execute("CREATE TABLE IF NOT EXISTS snapshot_file_acks (fingerprint TEXT PRIMARY KEY, event_id TEXT NOT NULL)")

    def _files_key(self, day, digests):
        names = {"account_snapshot.json", "daily_equity.json", "positions_snapshot.json", "trades_snapshot.json"}
        if (not isinstance(day, str) or len(day) != 8 or not day.isascii() or not day.isdigit()
                or set(digests) != names or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for v in digests.values())):
            raise ValueError("invalid snapshot file fingerprint")
        raw = json.dumps([1, self.source, self.strategy, self.account_id, day,
                          sorted(digests.items())], separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def files_sent(self, day, digests):
        key = self._files_key(day, digests)
        row = self.db.execute("SELECT event_id FROM snapshot_file_acks WHERE fingerprint=?", (key,)).fetchone()
        if row is None:
            return False
        if not self.db.execute("SELECT 1 FROM sent WHERE event_id=?", (row[0],)).fetchone():
            raise ValueError("snapshot ACK fingerprint lacks durable sent record")
        return True


    def send_batch(self, day: str, raw_files: dict[str, bytes], validate) -> bool:
        names = {"account_snapshot.json", "daily_equity.json",
                 "positions_snapshot.json", "trades_snapshot.json"}
        if not isinstance(day, str) or len(day) != 8 or not day.isdigit() or set(raw_files) != names:
            raise ValueError("invalid snapshot batch")
        if any(not isinstance(v, bytes) or len(v) > MAX_SNAPSHOT_PAYLOAD_BYTES for v in raw_files.values()):
            raise ValueError("invalid snapshot file bytes")
        digests = {name: hashlib.sha256(value).hexdigest() for name, value in raw_files.items()}
        if self.files_sent(day, digests):
            return False
        key = self._files_key(day, digests)
        payload = json.dumps({name: base64.b64encode(raw_files[name]).decode("ascii")
                              for name in sorted(names)}, separators=(",", ":")).encode()
        if len(payload) > MAX_SNAPSHOT_PAYLOAD_BYTES:
            raise ValueError("transport payload exceeds decoded size limit")
        digest = hashlib.sha256(payload).hexdigest()
        record_id = f"{day}:{digest}"
        event_id = hashlib.sha256(json.dumps([self.source, self.strategy, self.account_id,
                    "snapshot", record_id, digest], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if self.db.execute("SELECT 1 FROM sent WHERE event_id=?", (event_id,)).fetchone():
            with self.db:
                self.db.execute("INSERT INTO snapshot_file_acks VALUES (?, ?)", (key, event_id))
            return False
        validate(raw_files)
        envelope = Envelope.for_bytes(self.source, self.strategy, self.account_id,
                                      "snapshot", record_id, payload)
        self.sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=envelope.to_json(),
            MessageGroupId=f"{self.strategy}:{self.account_id}",
            MessageDeduplicationId=envelope.event_id,
        )
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sent VALUES (?)", (envelope.event_id,))
            self.db.execute("INSERT INTO snapshot_file_acks VALUES (?, ?)", (key, envelope.event_id))
        return True

    def close(self):
        self.db.close()


def snapshot_files(envelope: Envelope) -> dict[str, bytes]:
    if envelope.kind != "snapshot":
        raise ValueError("snapshot envelope required")
    try:
        doc = json.loads(envelope.payload)
        if set(doc) != {"account_snapshot.json", "daily_equity.json",
                        "positions_snapshot.json", "trades_snapshot.json"}:
            raise ValueError("incomplete snapshot batch")
        return {name: base64.b64decode(value, validate=True) for name, value in doc.items()}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid snapshot payload") from exc


class QueueConsumer:
    """At-least-once consumer; handler must use event_id for durable idempotency."""

    def __init__(self, state_path, sqs_client, queue_url, handler, *, required_version=None):
        if required_version is not None and (type(required_version) is not int or required_version not in (1,2)):raise ValueError('trusted protocol version required')
        self.required_version=required_version
        self.sqs, self.queue_url, self.handler = sqs_client, _identifier(queue_url), handler
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(state_path))
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS consumed (event_id TEXT PRIMARY KEY)")

    def poll(self) -> int:
        messages = self.sqs.receive_message(
            QueueUrl=self.queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=20,
            VisibilityTimeout=120,
        ).get("Messages", [])
        for message in messages:
            envelope = Envelope.from_json(message["Body"])
            # A durable pre-migration v1 identity may be redelivered after an
            # unknown DeleteMessage ACK. Historical dedup only, never a writer
            # fallback or permission for unknown old-protocol input.
            if not isinstance(envelope,SourceEnvelope) and self.db.execute("SELECT 1 FROM consumed WHERE event_id=?",(envelope.event_id,)).fetchone():
                self.sqs.delete_message(QueueUrl=self.queue_url,ReceiptHandle=message["ReceiptHandle"])
                continue
            if self.required_version is not None and (2 if isinstance(envelope,SourceEnvelope) else 1)!=self.required_version:
                raise ValueError("envelope differs from trusted source protocol version")
            # Archive registration must precede even an existing consumed check.
            # This is additive to old inbox state, without replaying business inputs.
            prepare = getattr(self.handler, "prepare", None)
            if prepare is not None:
                prepare(envelope)
            seen = self.db.execute("SELECT 1 FROM consumed WHERE event_id=?", (envelope.event_id,)).fetchone()
            if not seen:
                self.handler(envelope)
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO consumed VALUES (?)", (envelope.event_id,))
            self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=message["ReceiptHandle"])
        return len(messages)

    def close(self):
        self.db.close()
