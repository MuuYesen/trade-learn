"""Explicit send-only SQS configuration for Cerebro's optional monitor.

No platform token, configuration HTTP endpoint, default AWS identity chain or
credential file is used. Source bytes, outbox and proof-gated GC stay unchanged.
"""
from dataclasses import replace
import re
from urllib.parse import urlsplit

from .sender import BackgroundSender, require_sqs_dependencies


def _https(value, *, origin=False):
    if not isinstance(value, str) or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError('fixed HTTPS monitor destination required')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or origin and parsed.path not in ('', '/')):
        raise ValueError('fixed HTTPS monitor destination required')
    _ = parsed.port
    return parsed


class MonitorConnection:
    """Inert, per-run SQS settings; only a sender worker creates the AWS client."""
    def __init__(self, endpoint, credentials, receipt_origin, archive_binding_id):
        queue = _https(endpoint)
        match = re.fullmatch(r'sqs\.([a-z]{2}(?:-[a-z]+)+-[0-9])\.amazonaws\.com(?:\.cn)?', queue.netloc)
        if match is None or re.fullmatch(r'/[0-9]{12}/[A-Za-z0-9_-]{1,80}(?:\.fifo)?', queue.path) is None:
            raise ValueError('exact AWS SQS queue URL required')
        region = match.group(1)
        if queue.netloc.endswith('.com.cn') != region.startswith('cn-'):
            raise ValueError('SQS region and partition differ')
        if (not isinstance(credentials, dict) or not {'access_key_id', 'secret_access_key'} <= set(credentials)
                or set(credentials) - {'access_key_id', 'secret_access_key', 'session_token'}
                or any(not isinstance(v, str) or not v or len(v) > 4096
                       or any(ord(c) <= 32 or ord(c) == 127 for c in v) for v in credentials.values())):
            raise ValueError('explicit restricted AWS sending credentials required')
        _https(receipt_origin, origin=True)
        if not isinstance(archive_binding_id, str) or re.fullmatch('[A-Za-z0-9_-]{1,128}', archive_binding_id) is None:
            raise ValueError('trusted archive binding required')
        self.endpoint, self.region = endpoint, region
        self.receipt_origin, self.archive_binding_id = receipt_origin, archive_binding_id
        self._credentials = dict(credentials)
        self._client = None
        self._closed = False

    @classmethod
    def from_options(cls, options):
        fields = {'endpoint', 'credentials', 'receipt_origin', 'archive_binding_id'}
        if not isinstance(options, dict) or set(options) != fields:
            raise ValueError('monitor requires endpoint, credentials, receipt_origin and archive_binding_id; platform token is not SQS authorization')
        return cls(**options)

    def client(self):
        if self._closed:
            raise RuntimeError('monitor connection closed')
        if self._client is None:
            boto3, config = require_sqs_dependencies()
            keys = self._credentials
            self._client = boto3.client('sqs', region_name=self.region,
                aws_access_key_id=keys['access_key_id'], aws_secret_access_key=keys['secret_access_key'],
                aws_session_token=keys.get('session_token'),
                config=config(connect_timeout=5, read_timeout=10, retries={'max_attempts': 1}))
        return self._client

    def bind(self, settings):
        if self._closed:
            raise RuntimeError('monitor connection closed')
        require_sqs_dependencies()  # Fail before creating recorder/state or threads.
        bound = replace(settings, sender_config=self.endpoint,
                        receipt_origin=self.receipt_origin, archive_binding_id=self.archive_binding_id)
        def factory(current, *, flush):
            if current != bound:
                raise ValueError('monitor transport identity changed')
            return BackgroundSender(current, client_factory=self.client, queue_url=self.endpoint, flush=flush)
        return bound, factory

    def close(self):
        self._closed = True
        self._credentials.clear()
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
