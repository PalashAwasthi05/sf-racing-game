import json,urllib.request,zlib,gzip
from pathlib import Path
root=Path(__file__).resolve().parent/'imagery-v12'
d=json.loads((root/'selected.json').read_text());rec=next(r for r in d['images'] if r.get('is_pano') and r.get('sfm_cluster'))
for field in ('mesh','sfm_cluster'):
 target=root/(field+'-sample.bin')
 with urllib.request.urlopen(rec[field]['url'],timeout=50) as r:raw=r.read(80_000_000)
 target.write_bytes(raw)
 print(field,'image',rec['id'],'bytes',len(raw),'prefix',repr(raw[:30]))
 for kind,body in [('raw',raw),('zlib',None),('gzip',None)]:
  try:
   body=body if body is not None else zlib.decompress(raw) if kind=='zlib' else gzip.decompress(raw)
   j=json.loads(body);(root/(field+'-sample.json')).write_text(json.dumps(j));print('decode',kind,'top keys',list(j)[:15] if isinstance(j,dict) else 'array '+str(len(j)));break
  except Exception:pass
(root/'sample-image-id.txt').write_text(rec['id'])
