"""Authenticated registration discovery: fixed authority, queue scope and STS proof."""
import copy
import datetime as dt
import json
import time

import pytest

KEYS = {'access_key_id': 'AKIA'+'A'*16, 'secret_access_key': 'synthetic-secret'}
QUEUE = 'https://sqs.ap-southeast-1.amazonaws.com/123456789012/customer.fifo'


def challenge():
    return dict(protocol='leafquant.sts.v1', challenge_id='12345678-1234-4123-8123-123456789abc',
                expires_in=60, method='POST', url='https://sts.ap-southeast-1.amazonaws.com/',
                body='Action=GetCallerIdentity&Version=2011-06-15', region='ap-southeast-1',
                headers={'host': 'sts.ap-southeast-1.amazonaws.com',
                         'content-type': 'application/x-www-form-urlencoded; charset=utf-8',
                         'x-leafquant-context': 'a'*64})


def binding():
    return dict(protocol='leafquant.monitor-binding.v1', queue_url=QUEUE,
                receipt_origin='https://receipts.example.invalid', archive_binding_id='registered-binding',
                expires_at=int(time.time())+50)


class Response:
    status = 200
    def __init__(self, value): self.body=json.dumps(value).encode()
    def getheader(self, name, default=None): return 'application/json' if name.lower()=='content-type' else None
    def close(self): self.closed = True
    def read1(self, size):
        result, self.body=self.body[:size], self.body[size:]
        return result


def transport(replies, calls):
    class Connection:
        def __init__(self, host, **kwargs):
            assert host=='sentinel.leafquant.com' and kwargs['timeout']<=5
            calls.append(('connect', host))
            self.sock=type('Socket',(),{'settimeout':lambda self,value:calls.append(('timeout',value))})()
        def connect(self): pass
        def request(self, method, path, body, headers):
            assert method=='POST' and headers=={'Content-Type':'application/json','Accept':'application/json'}
            calls.append((path,json.loads(body)))
        def getresponse(self): return replies.pop(0)
        def close(self): calls.append(('close',))
    return Connection


def test_discovery_proves_key_possession_and_returns_only_verified_registration(monkeypatch):
    from leafquant_monitor import _monitor_binding as module
    calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection', transport([Response(challenge()),Response(binding())],calls))
    value=module.resolve(QUEUE,KEYS)
    assert value['archive_binding_id']=='registered-binding'
    requests=[item for item in calls if item[0].startswith('/api/')]
    assert requests[0]==('/api/monitor/challenge',{'audience':'leafquant-monitor','queue_url':QUEUE})
    request=requests[1]
    assert request[0]=='/api/monitor/binding'
    assert request[1]['scope']==requests[0][1]
    headers=request[1]['proof']['headers']
    assert headers['authorization'].startswith('AWS4-HMAC-SHA256 Credential='+KEYS['access_key_id']+'/')
    assert 'x-leafquant-context' in headers['authorization']
    assert 'synthetic-secret' not in json.dumps(calls)
    assert sum(c==('close',) for c in calls)==2


@pytest.mark.parametrize('mutate',[
    lambda c:c.update(url='https://foreign.invalid/'),
    lambda c:c.update(body='Action=AssumeRole'),
    lambda c:c.update(region='us-east-1'),
    lambda c:c.update(method='GET'),
    lambda c:c.update(expires_in=600),
    lambda c:c.update(challenge_id='not-uuid'),
    lambda c:c['headers'].update(host='foreign.invalid'),
    lambda c:c['headers'].update(extra='unsigned'),
    lambda c:c['headers'].update({'x-leafquant-context':'bad'}),
])
def test_untrusted_challenge_cannot_select_sts_request_or_send_proof(monkeypatch,mutate):
    from leafquant_monitor import _monitor_binding as module
    from leafquant_monitor.bootstrap import MonitorBindingUnavailable
    c=challenge();mutate(c);calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection',transport([Response(c)],calls))
    with pytest.raises(MonitorBindingUnavailable): module.resolve(QUEUE,KEYS)
    assert len([x for x in calls if len(x)==2 and x[0].startswith('/api')])==1


@pytest.mark.parametrize('change',[{'queue_url':QUEUE+'other'}, {'receipt_origin':'http://bad.invalid'},
    {'receipt_origin':'https://bad.invalid/redirect'}, {'archive_binding_id':'bad binding'},
    {'expires_at':int(time.time())-1},{'expires_at':int(time.time())+600}, {'expires_at':True},
    {'protocol':'other'}, {'extra':'not-registered'}])
def test_response_queue_origin_binding_expiry_and_shape_fail_closed(monkeypatch,change):
    from leafquant_monitor import _monitor_binding as module
    from leafquant_monitor.bootstrap import MonitorBindingUnavailable
    calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection',transport([Response(challenge()),Response({**binding(),**change})],calls))
    with pytest.raises(MonitorBindingUnavailable): module.resolve(QUEUE,KEYS)


@pytest.mark.parametrize('status',[301,302,307,400,403,429,503])
def test_http_failure_or_redirect_is_not_followed(monkeypatch,status):
    from leafquant_monitor import _monitor_binding as module
    from leafquant_monitor.bootstrap import MonitorBindingUnavailable
    response=Response({'error':'unavailable'});response.status=status;calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection',transport([response],calls))
    with pytest.raises(MonitorBindingUnavailable): module.resolve(QUEUE,KEYS)
    assert len([x for x in calls if x[0]=='connect'])==1


def test_internal_sts_transport_compatibility_is_not_removed():
    from leafquant_monitor.bootstrap import MonitorConnection
    value=MonitorConnection(QUEUE,{**KEYS,'session_token':'internal-only'},
                            'https://receipts.example.invalid','registered-binding')
    assert value._credentials['session_token']=='internal-only'
    value.close()


def test_transport_closes_rejected_response_and_sets_remaining_deadline(monkeypatch):
    from leafquant_monitor import _monitor_binding as module
    from leafquant_monitor.bootstrap import MonitorBindingUnavailable
    response=Response({'error':'denied'});response.status=403;calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection',transport([response],calls))
    with pytest.raises(MonitorBindingUnavailable): module.resolve(QUEUE,KEYS)
    assert getattr(response,'closed',False)
    timeouts=[item[1] for item in calls if item[0]=='timeout']
    assert len(timeouts)>=2 and all(0<value<=3 for value in timeouts)


def test_oversized_reply_cannot_escape_body_budget(monkeypatch):
    from leafquant_monitor import _monitor_binding as module
    from leafquant_monitor.bootstrap import MonitorBindingUnavailable
    response=Response({'padding':'x'*25000});calls=[]
    monkeypatch.setattr(module.http.client,'HTTPSConnection',transport([response],calls))
    with pytest.raises(MonitorBindingUnavailable): module.resolve(QUEUE,KEYS)
    assert getattr(response,'closed',False)
