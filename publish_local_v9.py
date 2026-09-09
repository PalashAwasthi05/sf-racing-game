"""Validate and expose the detailed review update; no network publishing."""
import gzip,hashlib,json,shutil,struct,zipfile,ast
from pathlib import Path
import numpy as np
import trimesh

root=Path(__file__).resolve().parents[1];work=root/'work';out=root/'outputs';public=work/'king-street/public';candidate=work/'detailed-v9'
report=json.loads((candidate/'report.json').read_text());raw=(candidate/'candidate.glb').read_bytes();digest=hashlib.sha256(raw).hexdigest()
assert raw[:4]==b'glTF' and struct.unpack_from('<I',raw,8)[0]==len(raw)
old=trimesh.load(out/'king-street-pilot.glb',force='scene',process=False);new=trimesh.load(candidate/'candidate.glb',force='scene',process=False)
changed={old.geometry[r['mesh']].metadata.get('featureId') for r in report['records'] if r['applied']}
unchanged=0
for name,m in old.geometry.items():
    if m.metadata.get('featureId') in changed:continue
    n=new.geometry[name]
    assert np.array_equal(m.faces,n.faces) and np.array_equal(m.vertices,n.vertices),name
    if hasattr(m.visual,'uv') and m.visual.uv is not None:assert np.array_equal(m.visual.uv,n.visual.uv),name
    unchanged+=1
for m in new.geometry.values():
    assert np.isfinite(m.vertices).all()
    if hasattr(m.visual,'uv') and m.visual.uv is not None:assert np.isfinite(m.visual.uv).all()
report['validation']={'unchangedUnrelatedMeshes':unchanged,'finitePositionsAndUVs':True,'roofResidualsAreNotImageAccuracy':True}
report['outputSha256']=digest;report['sourceSceneSha256']=hashlib.sha256((out/'king-street-pilot.glb').read_bytes()).hexdigest()
history=work/'history/before-v9-preview';history.mkdir(parents=True,exist_ok=True)
for name in ['model.json','world.json','generation.json']:
    if not (history/name).exists():shutil.copy2(public/name,history/name)
packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw
chunks=[]
for i,start in enumerate(range(0,len(packed),4_400_000)):
    name=f'world-{digest[:12]}-{i:02}.bin';(public/name).write_bytes(packed[start:start+4_400_000]);chunks.append('/'+name)
w=json.loads((history/'world.json').read_text());w['previewRevision']='STREET STUDY / ROOF REFINEMENT';w['updateSummary']='This update repairs two roof sections using measured slopes and brings their adjoining walls to the same roof surface. Other roof shapes, window styles, door layouts and many materials still need verification against street imagery.'
w['counts']['roofSectionsExamined']=report['capsAttempted'];w['counts']['roofSectionsRepaired']=report['capsReplaced']
(public/'world.json').write_text(json.dumps(w,separators=(',',':')))
generation={'pipelineVersion':'detailed-baseline-0.7 + general-roof-repair-0.9','allExistingRoofSectionsExamined':True,'fullFacadeRegeneration':False,'legacyFacadeGeometryPreserved':True,'roofSectionsExamined':report['capsAttempted'],'roofSectionsRepaired':report['capsReplaced'],'outputSha256':digest,'independentRealismGate':'NOT PASSED'}
(public/'generation.json').write_text(json.dumps(generation,separators=(',',':')))
(public/'model.json').write_text(json.dumps({'compression':'gzip','contentHash':digest[:12],'chunks':chunks,'compressedBytes':len(packed),'uncompressedBytes':len(raw)},separators=(',',':')))
(out/'world-building-update-report.json').write_text(json.dumps(report,indent=2));shutil.copy2(candidate/'candidate.glb',out/'world-building-update.glb')
(out/'world-building-update-release.json').write_text(json.dumps({**generation,'transferBytes':len(packed),'validation':report['validation'],'localPreview':'http://127.0.0.1:3001/','hostedSiteUpdated':False},indent=2))
print(json.dumps({'roofSectionsExamined':report['capsAttempted'],'roofSectionsRepaired':report['capsReplaced'],'unrelatedMeshesPreserved':unchanged,'transferBytes':len(packed),'outputSha256':digest}))
