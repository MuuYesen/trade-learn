import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[2])
from leafquant_monitor.source_file_sender import SourceFileSender
from leafquant_monitor.sqs_transport import Envelope
class Queue:
 def __init__(self): self.messages=[]
 def send_message(self,**kwargs):self.messages.append(kwargs['MessageBody'])
q=Queue(); d=Path(sys.argv[1]); s=SourceFileSender(d/'outbox.sqlite',q,'https://test.fifo','BYBIT','strategy','account','public-jsonl-v1',archive_binding_id='test')
expected=b''
for line in sys.stdin:
 cmd=line.strip()
 if cmd!='seal':
  i=int(cmd);expected+=(json.dumps({'n':i,'x':'x'*(40000 if i%10==0 else 3)},separators=(',',':'))+'\n').encode()
 for f in sorted(d.glob('*.log')):
  pub=json.loads(Path(str(f)+'.source.json').read_text())
  keys=('device','inode','published_length','publication_sequence','sealed','owner_generation')
  s.capture(f,f.stem,native_time_anchor=pub['native_time_anchor'],sealed=pub['sealed'],business_start=0,publication={k:pub[k] for k in keys})
 s.send_once()
 chunks=[Envelope.from_json(m) for m in q.messages]
 actual=b''.join(c.payload for c in chunks if c.kind=='source_chunk' and c.source_file['purpose']=='business')
 assert actual==expected,(cmd,len(actual),len(expected))
 if cmd=='seal':
  rows=s.db.execute('SELECT manifest FROM source_files ORDER BY file_id').fetchall()
  assert len(rows)>=2 and all(x[0] for x in rows)
  counts=[len(json.loads(x[0])['chunks']) for x in rows]
  assert max(counts)<=256,counts
  for (raw,) in rows:
   manifest=json.loads(raw);payload=(d/(manifest['file_id']+'.log')).read_bytes()
   assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
  print('OK chunks='+','.join(map(str,counts)),flush=True);break
 print('OK',flush=True)
s.close()
