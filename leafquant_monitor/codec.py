"""Shared bounded encoding for large AccountState.info JSON objects."""
import base64,copy,gzip,hashlib,json,zlib

KEY='leafquant_account_info_encoding_v1'
MAX_RAW=4*1024*1024
MAX_LINE=96*1024

def _bytes(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def _info(fact):
    if fact.get('kind')!='account_observation':return None
    fields=fact.get('fields')
    if not isinstance(fields,dict) or fields.get('scope') not in ('framework_account_event','framework_cache'):return None
    return fields.get('native_info')

def _encode_fact_and_json(fact, *, sort_keys=False):
    """Validate and serialize once; preserve the logger wire key order."""
    def render(value):
        return json.dumps(value,ensure_ascii=False,sort_keys=sort_keys,separators=(',',':'),allow_nan=False)
    info=_info(fact)
    if isinstance(info,dict) and KEY in info:
        decode_fact(fact)  # Same bounded integrity check for a native cache carrier.
        text=render(fact)
        if len(text.encode('utf-8'))+1>MAX_LINE:raise ValueError('compressed account fact exceeds log-line limit')
        return fact,text
    if not isinstance(info,dict):
        text=render(fact)
        if len(text.encode('utf-8'))+1>MAX_LINE:raise ValueError('oversize unencodable log fact')
        return fact,text
    # Serialize the native JSON once. Object key order carries no source facts;
    # canonical native info is also the exact established compression input.
    raw=_bytes(info)
    def object_text(value, override_key, override_text):
        items=sorted(value.items()) if sort_keys else value.items()
        parts=[]
        for key,child in items:
            # Preserve JSON's existing key coercion/escaping without a sentinel.
            prefix=render({key:None})[1:-5]
            parts.append(prefix+(override_text if key==override_key else render(child)))
        return '{'+','.join(parts)+'}'
    field_template=object_text(fact['fields'],'native_info','null')
    template=object_text(fact,'fields',field_template)
    if len(template.encode('utf-8'))-4+len(raw)+1<=MAX_LINE:
        fields_text=object_text(fact['fields'],'native_info',raw.decode('utf-8'))
        return fact,object_text(fact,'fields',fields_text)
    if len(raw)>MAX_RAW:raise ValueError('expanded account info budget exceeded')
    encoded=base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode('ascii')
    # Clone retained fields, never clone the large value being replaced.
    result={key:copy.deepcopy(value) for key,value in fact.items() if key!='fields'}
    result['fields']={key:copy.deepcopy(value) for key,value in fact['fields'].items() if key!='native_info'}
    result['fields']['native_info']={KEY:{'codec':'gzip-base64','raw_length':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'data':encoded}}
    text=render(result)
    if len(text.encode('utf-8'))+1>MAX_LINE:raise ValueError('compressed account fact exceeds log-line limit')
    return result,text

def encode_fact(fact):
    return _encode_fact_and_json(fact,sort_keys=True)[0]

def _unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key in account info')
        out[k]=v
    return out

def decode_fact(fact):
    info=_info(fact)
    if not isinstance(info,dict) or KEY not in info:return fact
    if set(info)!={KEY}:raise ValueError('ambiguous encoded account info')
    obj=info[KEY]
    if not isinstance(obj,dict) or set(obj)!={'codec','raw_length','sha256','data'} or obj['codec']!='gzip-base64':raise ValueError('invalid encoding header')
    size=obj['raw_length'];sha=obj['sha256'];data=obj['data']
    if (type(size) is not int or not 0<=size<=MAX_RAW or not isinstance(sha,str) or len(sha)!=64
            or any(c not in '0123456789abcdef' for c in sha) or not isinstance(data,str) or len(data)>MAX_LINE):raise ValueError('invalid encoding limits')
    try:
        packed=base64.b64decode(data,validate=True)
        if base64.b64encode(packed).decode('ascii')!=data:raise ValueError('noncanonical base64')
        dec=zlib.decompressobj(16+zlib.MAX_WBITS);raw=dec.decompress(packed,size+1)
        if len(raw)!=size or not dec.eof or dec.unused_data or dec.unconsumed_tail:raise ValueError('truncated, concatenated or excessive gzip')
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('account info hash mismatch')
        value=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (ValueError,TypeError,UnicodeError,zlib.error,RecursionError) as e:raise ValueError('invalid encoded account info') from e
    if not isinstance(value,dict) or KEY in value:raise ValueError('invalid decoded account info')
    result=copy.deepcopy(fact);result['fields']['native_info']=value;return result


def decoded_info(info):
    """Logical account-info identity across plain and existing gzip carriers."""
    return decode_fact({'kind':'account_observation','fields':{'scope':'framework_account_event','native_info':info}})['fields']['native_info']
