"""Direct SQS assembly with synthetic sending credentials and real source wire."""
from dataclasses import replace
import time

import pytest


def options():
    return dict(endpoint='https://sqs.ap-southeast-1.amazonaws.com/123456789012/customer.fifo',
        receipt_origin='https://receipts.example.invalid', archive_binding_id='synthetic-binding',
        credentials=dict(access_key_id='synthetic-key', secret_access_key='synthetic-secret'))


def test_connection_is_inert_and_unknown_platform_token_is_rejected(tmp_path, monkeypatch):
    import leafquant_monitor.bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', lambda: pytest.fail('inert constructor loaded AWS SDK'))
    value = bootstrap.MonitorConnection(**options())
    assert value.region == 'ap-southeast-1'
    assert 'synthetic-secret' not in repr(value)
    assert value._client is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match='platform token'):
        bootstrap.MonitorConnection.from_options({'endpoint': options()['endpoint'], 'token': 'a' * 40})


@pytest.mark.parametrize('change', [dict(endpoint='http://sqs.ap-southeast-1.amazonaws.com/123456789012/q'),
    dict(endpoint='https://foreign.invalid/q'), dict(endpoint=options()['endpoint']+'?token=x'),
    dict(endpoint='https://user:secret@sqs.ap-southeast-1.amazonaws.com/123456789012/q'),
    dict(credentials={'access_key_id': 'x'}), dict(credentials={'access_key_id': 'x', 'secret_access_key': 'a\nb'}),
    dict(receipt_origin='http://receipt.invalid'), dict(archive_binding_id='invalid binding'), dict(extra=True)])
def test_invalid_or_broadened_configuration_rejected(change):
    from leafquant_monitor.bootstrap import MonitorConnection
    with pytest.raises(ValueError):
        MonitorConnection.from_options(dict(options(), **change))


def test_missing_sdk_rejects_before_state_or_client(tmp_path, monkeypatch):
    import leafquant_monitor.bootstrap as bootstrap
    from leafquant_monitor.logger import MonitorSettings
    def missing():
        raise ImportError('missing optional SDK')
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', missing)
    value = bootstrap.MonitorConnection(**options())
    with pytest.raises(ImportError):
        value.bind(MonitorSettings('TEST', 's', 'a', str(tmp_path/'state')))
    assert not (tmp_path/'state').exists()


def test_bound_sender_uses_explicit_credentials_in_memory_and_pins_identity(tmp_path, monkeypatch):
    import leafquant_monitor.bootstrap as bootstrap
    from leafquant_monitor.logger import MonitorSettings
    calls = []
    class Client:
        def close(self):calls.append('close')
    class Boto:
        def client(self, name, **kwargs):
            calls.append((name, kwargs));return Client()
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', lambda: (Boto(), lambda **kw: kw))
    value = bootstrap.MonitorConnection(**options())
    bound, sender_factory = value.bind(MonitorSettings('TEST', 's', 'a', str(tmp_path/'state')))
    sender = sender_factory(bound, flush=lambda: None)
    assert sender.thread is None and calls == []
    sender.factory()
    assert calls[0][0] == 'sqs'
    assert calls[0][1]['aws_access_key_id'] == 'synthetic-key'
    assert calls[0][1]['aws_secret_access_key'] == 'synthetic-secret'
    assert calls[0][1]['region_name'] == 'ap-southeast-1'
    assert sender.queue_url == options()['endpoint']
    assert not (tmp_path/'state').exists()
    with pytest.raises(ValueError):sender_factory(replace(bound, account_id='foreign'), flush=lambda: None)
    value.close()
    assert calls[-1] == 'close' and value._credentials == {}
    with pytest.raises(RuntimeError, match='closed'):value.client()


def test_native_log_reaches_existing_source_wire_without_receipt_deletion(tmp_path, monkeypatch):
    pytest.importorskip('tradelearn._rust')
    import leafquant_monitor.bootstrap as bootstrap
    from leafquant_monitor.logger import MonitorSettings, TradeLogger
    from leafquant_monitor.archive_receipt_client import ArchiveReceiptClient
    from leafquant_monitor.sqs_transport import Envelope
    bodies = []
    class Sqs:
        def send_message(self, **kwargs):
            assert kwargs['QueueUrl'] == options()['endpoint']
            bodies.append(kwargs['MessageBody'])
            return {'MessageId': 'synthetic-ack'}
        def close(self):pass
    class Boto:
        def client(self, name, **kwargs):return Sqs()
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', lambda: (Boto(), lambda **kw: kw))
    monkeypatch.setattr(ArchiveReceiptClient, 'fetch', lambda *args: None)
    remote = bootstrap.MonitorConnection(**options())
    bound, factory = remote.bind(MonitorSettings('TEST', 'strategy', 'account', str(tmp_path/'logs')))
    from tests.windows_private_directory import make_private_test_directory
    make_private_test_directory(tmp_path)
    logger = TradeLogger(bound, sender_factory=factory)
    try:
        logger.sender.interval = 0.005
        logger.record('lifecycle', {'state': 'synthetic-started'})
        logger.flush()
        deadline = time.monotonic() + 2
        while not bodies and time.monotonic() < deadline:time.sleep(0.005)
        assert bodies, logger.sender.health()
        assert any(b'synthetic-started' in Envelope.from_json(body).payload for body in bodies)
        assert all('synthetic-secret' not in body for body in bodies)
        assert list((tmp_path/'logs/segments').glob('*.log'))
    finally:
        logger.finish_delivery(timeout=3)
        remote.close()


def test_public_monitor_is_only_endpoint_and_long_lived_key_pair():
    from leafquant_monitor.bootstrap import MonitorConnection
    public = {k: v for k, v in options().items() if k in ('endpoint', 'credentials')}
    value = MonitorConnection.from_options(public)
    assert value.receipt_origin is None and value.archive_binding_id is None
    assert value._client is None
    for extra in ({'receipt_origin': options()['receipt_origin']},
                  {'archive_binding_id': 'synthetic-binding'}):
        with pytest.raises(ValueError):
            MonitorConnection.from_options({**public, **extra})
    with pytest.raises(ValueError):
        MonitorConnection.from_options({**public, 'credentials': {**public['credentials'], 'session_token': 'temporary'}})


def test_unresolved_registration_never_constructs_sender_or_uses_aws(tmp_path, monkeypatch):
    import leafquant_monitor.bootstrap as bootstrap
    from leafquant_monitor.logger import MonitorSettings
    monkeypatch.setattr(bootstrap, 'require_sqs_dependencies', lambda: pytest.fail('unverified binding used AWS'))
    public = {k: v for k, v in options().items() if k in ('endpoint', 'credentials')}
    value = bootstrap.MonitorConnection.from_options(public)
    with pytest.raises(bootstrap.MonitorBindingUnavailable, match='trusted'):
        value.bind(MonitorSettings('TEST', 's', 'a', str(tmp_path/'state')))
    assert value._client is None and not (tmp_path/'state').exists()
