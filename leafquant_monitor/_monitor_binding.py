"""Private monitor registration discovery using the registered IAM user's proof.

Only the fixed Sentinel TLS authority receives a locally signed fixed STS
request. No secret, platform bearer, arbitrary URL, AWS permission or account
identity is provisioned here. An unverified response never configures a sender.
"""
import datetime as dt
import hashlib
import hmac
import http.client
import json
import re
import ssl
import time

_AUTHORITY = 'sentinel.leafquant.com'
_STS_HOST = 'sts.ap-southeast-1.amazonaws.com'
_STS_BODY = 'Action=GetCallerIdentity&Version=2011-06-15'
_LIMIT = 24576


def _deny():
    from .bootstrap import MonitorBindingUnavailable
    raise MonitorBindingUnavailable('trusted monitor archive registration unavailable; local logs retained, remote delivery disabled')


def _post(path, value, deadline):
    # http.client neither follows redirects nor consumes environment proxies.
    def remaining():
        seconds = deadline-time.monotonic()
        if seconds <= 0:
            _deny()
        return min(3, seconds)
    connection = http.client.HTTPSConnection(_AUTHORITY, timeout=remaining(),
                                            context=ssl.create_default_context())
    response = None
    try:
        connection.connect()
        # Keep the socket while a Connection: close response still owns its
        # file handle. Reapply the *remaining* total budget before every read.
        sock = connection.sock
        sock.settimeout(remaining())
        connection.request('POST', path, body=json.dumps(value,separators=(',',':')),
                           headers={'Content-Type':'application/json','Accept':'application/json'})
        sock.settimeout(remaining())
        response = connection.getresponse()
        if response.status != 200 or response.getheader('Content-Type','').split(';')[0].strip().lower() != 'application/json':
            _deny()
        body = bytearray()
        while True:
            sock.settimeout(remaining())
            chunk = response.read1(min(4096,_LIMIT+1-len(body)))
            if not chunk:
                break
            body.extend(chunk)
            if len(body)>_LIMIT:
                _deny()
        if time.monotonic() >= deadline:
            _deny()
        result = json.loads(body)
        if not isinstance(result,dict):
            _deny()
        return result
    finally:
        if response is not None:
            response.close()
        connection.close()


def _sign(credentials, challenge):
    if (not isinstance(credentials,dict) or set(credentials)!={'access_key_id','secret_access_key'}
            or not re.fullmatch(r'AKIA[A-Z0-9]{16}',str(credentials.get('access_key_id','')))
            or any(not isinstance(v,str) or not v or len(v)>4096 or re.search(r'[\x00-\x20\x7f]',v) for v in credentials.values())):
        _deny()
    if (not isinstance(challenge,dict) or set(challenge)!={'protocol','challenge_id','expires_in','method','url','body','region','headers'}
            or challenge['protocol']!='leafquant.sts.v1' or type(challenge['expires_in']) is not int or challenge['expires_in']!=60
            or not re.fullmatch(r'[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}',str(challenge['challenge_id']))
            or challenge['method']!='POST' or challenge['url']!='https://'+_STS_HOST+'/'
            or challenge['body']!=_STS_BODY or challenge['region']!='ap-southeast-1'):
        _deny()
    headers=challenge['headers']
    if (not isinstance(headers,dict) or set(headers)!={'host','content-type','x-leafquant-context'}
            or headers['host']!=_STS_HOST or headers['content-type']!='application/x-www-form-urlencoded; charset=utf-8'
            or not re.fullmatch(r'[a-f0-9]{64}',str(headers['x-leafquant-context']))):
        _deny()
    date=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    headers={**headers,'x-amz-date':date}
    names=';'.join(sorted(headers))
    canonical='POST\n/\n\n'+''.join(k+':'+headers[k]+'\n' for k in sorted(headers))+'\n'+names+'\n'+hashlib.sha256(_STS_BODY.encode()).hexdigest()
    scope=date[:8]+'/ap-southeast-1/sts/aws4_request'
    message='AWS4-HMAC-SHA256\n'+date+'\n'+scope+'\n'+hashlib.sha256(canonical.encode()).hexdigest()
    key=('AWS4'+credentials['secret_access_key']).encode()
    for part in (date[:8],'ap-southeast-1','sts','aws4_request'):
        key=hmac.new(key,part.encode(),hashlib.sha256).digest()
    signature=hmac.new(key,message.encode(),hashlib.sha256).hexdigest()
    headers['authorization']='AWS4-HMAC-SHA256 Credential='+credentials['access_key_id']+'/'+scope+', SignedHeaders='+names+', Signature='+signature
    return {'challenge_id':challenge['challenge_id'],'headers':headers}


def resolve(queue_url, credentials):
    """Return a short-lived verified registration, or retain local-only operation."""
    from .bootstrap import MonitorBindingUnavailable, _https
    try:
        # The public constructor already checks exact AWS queue syntax. Reject
        # non-registered key kinds before requesting a challenge as well.
        if not re.fullmatch(r'AKIA[A-Z0-9]{16}',str(credentials.get('access_key_id',''))):
            _deny()
        if not re.fullmatch(r'https://sqs\.ap-southeast-1\.amazonaws\.com/[0-9]{12}/[A-Za-z0-9_-]{1,80}(?:\.fifo)?',queue_url):
            _deny()
        deadline=time.monotonic()+5
        scope={'audience':'leafquant-monitor','queue_url':queue_url}
        challenge=_post('/api/monitor/challenge',scope,deadline)
        proof=_sign(credentials,challenge)
        result=_post('/api/monitor/binding',{'scope':scope,'proof':proof},deadline)
        now=time.time()
        if (set(result)!={'protocol','queue_url','receipt_origin','archive_binding_id','expires_at'}
                or result['protocol']!='leafquant.monitor-binding.v1' or result['queue_url']!=queue_url
                or type(result['expires_at']) is not int or not now<result['expires_at']<=now+60
                or not isinstance(result['archive_binding_id'],str)
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',result['archive_binding_id'])):
            _deny()
        _https(result['receipt_origin'],origin=True)
        return result
    except MonitorBindingUnavailable:
        raise
    except Exception:
        # Neither signed headers nor connection/response text reach diagnostics.
        _deny()
