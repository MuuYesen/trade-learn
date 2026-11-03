"""Proof-only pinned HTTPS origin client. No redirects, AWS or source deletion."""
import base64
import http.client
import json
import re
import ssl
from urllib.parse import urlsplit


class ArchiveReceiptClient:
    def __init__(self,origin,*,connection_factory=None):
        parsed=urlsplit(origin)
        if (parsed.scheme!='https' or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('','/')
                or parsed.query or parsed.fragment):
            raise ValueError('fixed trusted HTTPS receipt origin required')
        self.host,self.port=parsed.hostname,parsed.port or 443
        self.factory=connection_factory or http.client.HTTPSConnection
        self.context=ssl.create_default_context()

    def fetch(self,seal_event_id,token):
        if not isinstance(seal_event_id,str) or re.fullmatch('[0-9a-f]{64}',seal_event_id) is None:
            raise ValueError('exact seal event id required')
        if not isinstance(token,str) or re.fullmatch('[A-Za-z0-9_-]{43}',token) is None:
            raise ValueError('canonical receipt capability required')
        raw=base64.b64decode(token+'=',altchars=b'-_',validate=True)
        if len(raw)!=32 or base64.urlsafe_b64encode(raw).decode().rstrip('=')!=token:
            raise ValueError('canonical receipt capability required')
        connection=self.factory(self.host,self.port,timeout=5,context=self.context)
        try:
            connection.request('GET','/v1/archive-receipts/'+seal_event_id,
                               headers={'Authorization':'Bearer '+token,'Accept':'application/json'})
            response=connection.getresponse()
            if response.status in (202,404,429,503): return None
            if response.status!=200: raise ValueError('receipt endpoint rejected request')
            body=response.read(4097)
            if len(body)>4096: raise ValueError('receipt response exceeds budget')
            value=json.loads(body)
            if not isinstance(value,dict): raise ValueError('invalid receipt response')
            return value
        finally:
            connection.close()
