"""Validate retained materials/images and generated object exclusions before publishing."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from shapely.geometry import Point,shape
from replace_scene_layers_v21 import _read_glb

def image_hash(d,i):
 im=d.json['images'][i];v=d.json['bufferViews'][im['bufferView']];s=v.get('byteOffset',0)
 return hashlib.sha256(d.binary[s:s+v['byteLength']]).hexdigest()
def texture(d,i):
 t=dict(d.json['textures'][i]);t['source']=image_hash(d,t['source'])
 if 'sampler'in t:t['sampler']=d.json['samplers'][t['sampler']]
 return t
def normalize(d,v):
 if isinstance(v,list):return [normalize(d,x) for x in v]
 if not isinstance(v,dict):return v
 return {k:({**x,'index':texture(d,x['index'])} if k.lower().endswith('texture') and isinstance(x,dict) and 'index'in x else normalize(d,x)) for k,x in v.items()}
def signature(d,mesh):
 return [normalize(d,d.json['materials'][p['material']]) if 'material'in p else None for p in mesh['primitives']]
def validate(candidate):
 r=json.loads((candidate/'final/report.json').read_text());base=_read_glb(r['base']['path']);new=_read_glb(candidate/'final/candidate.glb');layers=set(r['layers'])
 assert hashlib.sha256((candidate/'final/candidate.glb').read_bytes()).hexdigest()==r['output']['sha256']
 assert r['preservationHashCheck']
 nm={m['name']:m for m in new.json['meshes']};assert len(nm)==len(new.json['meshes']);compared=0
 for m in base.json['meshes']:
  if m.get('extras',{}).get('layer') in layers:continue
  assert signature(base,m)==signature(new,nm[m['name']]),m['name'];compared+=1
 p=json.loads((candidate/'street-space-audit.json').read_text());s=json.loads((candidate/'street-space-geometry.json').read_text());forbidden=shape(s['forbidden']);accepted=[x for x in p['placements'] if 'position'in x]
 conflicts=[x for x in accepted if Point(x['position']).buffer(x['radiusM']).intersection(forbidden).area>1e-5]
 assert not conflicts,'Furniture intersects forbidden street space'
 result={'preservedMaterialAndImageComparisons':compared,'preservedGeometryChecks':len(r['preservationHashes']),'acceptedPlacements':len(accepted),'forbiddenFootprintIntersections':len(conflicts),'counts':dict(Counter(x['kind'] for x in accepted))}
 (candidate/'final/validation.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);a=ap.parse_args();print(json.dumps(validate(a.candidate)))
