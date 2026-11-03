"""Read durable native prefixes and retire only receipt-proved SDK sidecars."""
import json
import os
from pathlib import Path
import re
import stat

from .source_file_contract import validate_native_time_anchor


_NAME = re.compile(r'([a-f0-9]{32})-[0-9]{20}\.log')
_FIELDS = {'schema_version', 'path', 'owner_generation', 'device', 'inode',
           'published_length', 'publication_sequence', 'sealed', 'native_time_anchor'}
_CAPTURE = ('device', 'inode', 'published_length', 'publication_sequence', 'sealed', 'owner_generation')


def _private_json(path, limit=4096):
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600 or before.st_uid != os.getuid()
            or not 0 < before.st_size <= limit):
        raise ValueError('private bounded source metadata required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: tuple(getattr(value, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    if identity(before) != identity(opened) or identity(opened) != identity(after) or len(raw) != before.st_size:
        raise ValueError('source metadata changed during read')
    return json.loads(raw), identity(before)


def _validated(document, path, anchor):
    match = _NAME.fullmatch(path.name)
    if (not match or not isinstance(document, dict) or set(document) != _FIELDS
            or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['path'] != path.name or document['owner_generation'] != match[1]
            or type(document['sealed']) is not bool
            or any(type(document[k]) is not int or document[k] < 0 for k in ('device', 'inode', 'published_length'))
            or type(document['publication_sequence']) is not int or document['publication_sequence'] < 1
            or validate_native_time_anchor(document['native_time_anchor']) != validate_native_time_anchor(anchor)):
        raise ValueError('source publication binding differs')
    return {k: document[k] for k in _CAPTURE}


def source_publication(path, anchor):
    """None denotes an old sealed source; new active sources always have a marker."""
    path = Path(path)
    sidecar = path.with_name(path.name + '.source.json')
    if not os.path.lexists(sidecar):
        if os.path.lexists(path.with_suffix('.window')):
            raise ValueError('active source publication unavailable')
        return None
    document, _ = _private_json(sidecar)
    value = _validated(document, path, anchor)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (value['device'], value['inode'])
            or info.st_size < value['published_length']
            or (value['sealed'] and info.st_size != value['published_length'])):
        raise ValueError('source differs from durable publication')
    return value


def retire_source_sidecars(outbox, directory, *, limit=16, scan=64):
    """Bounded fair metadata GC; raw bytes must already be retired by the SDK."""
    if not 1 <= limit <= 16 or not 1 <= scan <= 64:
        raise ValueError('bounded source metadata retirement required')
    directory = Path(directory).absolute()
    key = 'sdk-sidecar-gc:' + outbox.producer_id
    found = outbox.db.execute('SELECT value FROM file_sender_meta WHERE key=?', (key,)).fetchone()
    cursor = found[0] if found else ''
    query = 'SELECT file_id,path,device,inode,verified FROM source_files WHERE producer=? AND file_id>? ORDER BY file_id LIMIT ?'
    rows = outbox.db.execute(query, (outbox.producer_id, cursor, scan)).fetchall()
    if not rows and cursor:
        rows = outbox.db.execute(query, (outbox.producer_id, '', scan)).fetchall()
    removed = 0
    scanned = 0
    error = False
    advanced = cursor
    for file_id, stored_path, device, inode, verified in rows:
        advanced = file_id
        scanned += 1
        try:
            path = directory / (file_id + '.log')
            if verified != 2 or stored_path != str(path) or not _NAME.fullmatch(path.name):
                continue
            if any(os.path.lexists(p) for p in (path, path.with_suffix('.pending'), path.with_suffix('.window'))):
                continue
            metadata = [path.with_name(path.name + '.source.json'), path.with_suffix('.native-time.json')]
            present = [p for p in metadata if os.path.lexists(p)]
            if not present:
                continue
            proof = outbox.verified_proof(file_id)
            if proof is None:
                continue
            anchor_row = outbox.db.execute('SELECT body FROM source_time_anchors WHERE producer=? AND file_id=?', (outbox.producer_id, file_id)).fetchone()
            if anchor_row is None:
                raise ValueError('retired source clock evidence unavailable')
            anchor = validate_native_time_anchor(json.loads(anchor_row[0]))
            checks = []
            for item in present:
                value, identity = _private_json(item)
                if item == metadata[0]:
                    publication = _validated(value, path, anchor)
                    expected = outbox.db.execute('SELECT body FROM source_publications WHERE producer=? AND file_id=?', (outbox.producer_id, file_id)).fetchone()
                    if (not publication['sealed'] or publication['published_length'] != proof['length']
                            or (publication['device'], publication['inode']) != (device, inode)
                            or expected is None or json.loads(expected[0]) != publication):
                        raise ValueError('retired publication differs from proved source')
                elif validate_native_time_anchor(value) != anchor:
                    raise ValueError('retired source clock differs')
                checks.append((item, identity))
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0))
            try:
                os.fsync(fd)
                for item, identity in checks:
                    current = os.stat(item.name, dir_fd=fd, follow_symlinks=False)
                    if tuple(getattr(current, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) != identity:
                        raise ValueError('source sidecar changed before retirement')
                if any(os.path.lexists(p) for p in (path, path.with_suffix('.pending'), path.with_suffix('.window'))):
                    raise ValueError('source reappeared before metadata retirement')
                for item, _ in checks:
                    os.unlink(item.name, dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            removed += 1
            if removed >= limit:
                break
        except (OSError, ValueError, TypeError):
            error = True
    if rows:
        outbox.db.execute('INSERT OR REPLACE INTO file_sender_meta VALUES(?,?)', (key, advanced))
    return {'removed': removed, 'error': error, 'scanned': scanned}
