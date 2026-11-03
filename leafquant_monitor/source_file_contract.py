"""Single sealed-file wire/proof contract; no AWS, network or filesystem access."""
import hashlib
import json
import re

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 256 * 1024
MAX_CHUNKS = 256
MAX_MANIFEST_BYTES = 256 * 1024
_HEX = re.compile(r'[0-9a-f]{64}')
_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,1023}')


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _token(value):
    return isinstance(value, str) and _TOKEN.fullmatch(value) and '..' not in value


def _sha(value):
    return isinstance(value, str) and _HEX.fullmatch(value)


def _validated_body(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {
            'schema_version', 'producer_id', 'generation', 'file_id', 'revision', 'length', 'sha256', 'chunks'}
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or any(not _token(manifest[k]) for k in ('producer_id', 'generation', 'file_id'))
            or not _integer(manifest['revision'], 0, 2**63-1)
            or not _integer(manifest['length'], 0, MAX_FILE_BYTES)
            or not _sha(manifest['sha256']) or not isinstance(manifest['chunks'], list)
            or not 0 <= len(manifest['chunks']) <= MAX_CHUNKS):
        raise ValueError('invalid sealed file manifest')
    # Validate lengths before serializing arbitrary metadata.
    cursor, records = 0, set()
    for chunk in manifest['chunks']:
        if (not isinstance(chunk, dict) or set(chunk) != {'offset', 'length', 'sha256', 'record_id'}
                or not _integer(chunk['offset'], 0, MAX_FILE_BYTES)
                or chunk['offset'] != cursor
                or not _integer(chunk['length'], 1, MAX_CHUNK_BYTES)
                or not _sha(chunk['sha256']) or not _token(chunk['record_id'])
                or chunk['record_id'] in records):
            raise ValueError('invalid sealed file coverage')
        cursor += chunk['length']
        records.add(chunk['record_id'])
        if cursor > manifest['length']:
            raise ValueError('sealed file range exceeds length')
    if cursor != manifest['length']:
        raise ValueError('incomplete sealed file coverage')
    body = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(body) > MAX_MANIFEST_BYTES:
        raise ValueError('sealed file manifest exceeds budget')
    return body


def source_manifest_digest(manifest):
    return hashlib.sha256(_validated_body(manifest)).hexdigest()



def validate_source_manifest(manifest):
    """Return a bounded canonical copy, rejecting gaps, overlap and extra fields."""
    return json.loads(_validated_body(manifest))


def validate_chunk_claim(claim, payload):
    fields = {'producer_id','generation','file_id','revision','offset','length','sha256','record_id','purpose'}
    if (not isinstance(claim, dict) or set(claim) != fields
            or any(not _token(claim[k]) for k in ('producer_id','generation','file_id','record_id'))
            or not _integer(claim['revision'], 0, 2**63-1)
            or not _integer(claim['offset'], 0, MAX_FILE_BYTES)
            or not _integer(claim['length'], 1, MAX_CHUNK_BYTES)
            or claim['offset'] + claim['length'] > MAX_FILE_BYTES
            or not _sha(claim['sha256']) or claim['purpose'] not in ('business','archive_only')
            or not isinstance(payload, bytes) or len(payload) != claim['length']
            or hashlib.sha256(payload).hexdigest() != claim['sha256']):
        raise ValueError('invalid source file chunk')
    return dict(claim)


def validate_source_receipt_shape(receipt):
    fields={'schema_version','binding_id','producer_id','generation','file_id','revision','length','sha256','manifest_sha256','status'}
    if (not isinstance(receipt,dict) or set(receipt)!=fields
            or type(receipt['schema_version']) is not int or receipt['schema_version']!=1
            or not isinstance(receipt['binding_id'],str) or re.fullmatch('[A-Za-z0-9_-]{1,128}',receipt['binding_id']) is None
            or receipt['status']!='verified'
            or any(not _token(receipt[k]) for k in ('producer_id','generation','file_id'))
            or not _integer(receipt['revision'],0,2**63-1)
            or not _integer(receipt['length'],0,MAX_FILE_BYTES)
            or not _sha(receipt['sha256']) or not _sha(receipt['manifest_sha256'])):
        raise ValueError('invalid source archive proof')
    return dict(receipt)


def validate_source_receipt(receipt, manifest, *, expected_binding_id):
    """Authenticate transport separately; then require exact local seal equality.

    Only call for a response from the pinned HTTPS management endpoint, with the
    per-file capability supplied in Authorization, never from arbitrary input.
    """
    receipt = validate_source_receipt_shape(receipt)
    if (not isinstance(expected_binding_id,str) or re.fullmatch('[A-Za-z0-9_-]{1,128}',expected_binding_id) is None
            or receipt['binding_id']!=expected_binding_id):
        raise ValueError('source archive receipt binding differs from trusted registration')
    checked = validate_source_manifest(manifest)
    expected = dict(schema_version=1, binding_id=expected_binding_id,status='verified',
                    **{k: checked[k] for k in ('producer_id','generation','file_id','revision','length','sha256')},
                    manifest_sha256=source_manifest_digest(checked))
    if (not isinstance(receipt, dict) or set(receipt) != set(expected)
            or type(receipt.get('schema_version')) is not int
            or type(receipt.get('length')) is not int or type(receipt.get('revision')) is not int
            or receipt != expected):
        raise ValueError('source archive receipt does not match local seal')
    return dict(expected)


SOURCE_FORMATS={'jsonl':'log_jsonl','log_text':'log_text','json':'json','binary':'binary',
                **{f'{source}_native_{role}':'json' for source in ('ctrader','qmt','ibkr','bybit') for role in ('snapshot','history')}}


def storage_content_kind(registered_format):
    if not isinstance(registered_format,str) or registered_format not in SOURCE_FORMATS:
        raise ValueError('unknown trusted source producer format')
    return SOURCE_FORMATS[registered_format]


def validate_native_time_anchor(anchor):
    """Producer-persisted UTC nanoseconds; never a receiver clock or client TZ."""
    if (not isinstance(anchor,dict) or set(anchor)!={'schema_version','start_ts_ns','end_ts_ns','source_schema'}
            or type(anchor['schema_version']) is not int or anchor['schema_version']!=1
            or not _integer(anchor['start_ts_ns'],1,2**63-1)
            or (anchor['end_ts_ns'] is not None and not _integer(anchor['end_ts_ns'],anchor['start_ts_ns'],2**63-1))
            or anchor['source_schema'] not in {'native-log-file.v1','account-evidence-observed.v1','collector-commit-observed.v1','ctrader-market-tape-opened.v1'}):
        raise ValueError('invalid producer native time anchor')
    return dict(anchor)
