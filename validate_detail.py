import json, pathlib, struct, gzip, hashlib
import numpy as np,trimesh
root=pathlib.Path(__file__).parent.parent;public=root/'work/king-street/public';raw=(root/'outputs/king-street-pilot.glb').read_bytes();header=json.loads(raw[20:20+struct.unpack_from('<I',raw,12)[0]])
manifest=json.load(open(public/'model.json'));packed=b''.join((public/p.lstrip('/')).read_bytes() for p in manifest['chunks']);assert gzip.decompress(packed)==raw;assert all((public/p.lstrip('/')).stat().st_size<=4*1024*1024 for p in manifest['chunks'])
s=trimesh.load(root/'outputs/king-street-pilot.glb',force='scene');finite=all(np.isfinite(m.vertices).all() and np.isfinite(m.visual.uv).all() for m in s.geometry.values());assert finite
assert all(m.faces.min()>=0 and m.faces.max()<len(m.vertices) for m in s.geometry.values())
w=json.load(open(public/'world.json'));assert len(w['buildings'])>0;assert len({b['id'] for b in w['buildings']})==len(w['buildings'])
e=json.load(open(root/'outputs/king-street-evidence.json'));features={b['id']:b for b in w['buildings']}
for wall in e['facadeWalls']:
 if wall.get('grammarId'):assert wall['heightM']>=features[wall['buildingId']]['height']*.60,wall
assert w.get('streetViews')
for m in s.geometry.values():
 assert m.visual.material.roughnessFactor is not None
result={'rawBytes':len(raw),'transferBytes':len(packed),'transferChunks':len(manifest['chunks']),'meshes':len(header['meshes']),'materials':len(header['materials']),'images':len(header['images']),'triangles':sum(header['accessors'][p['indices']]['count']//3 for m in header['meshes'] for p in m['primitives']),'finiteCoordinatesAndUVs':finite,'validIndices':True,'transferRoundtripMatches':True,'sha256':hashlib.sha256(raw).hexdigest(),'browserInteractionTested':False,'independentFacadeAccuracyVerified':False}
(root/'outputs/king-street-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
