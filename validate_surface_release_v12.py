import json,hashlib
from xml.etree import ElementTree
from pathlib import Path
import trimesh,numpy as np
p=Path('work/surface-final-v12');s=trimesh.load(p/'candidate.glb',force='scene',process=False);old=trimesh.load('work/detailed-v11/candidate.glb',force='scene',process=False)
finite=all(np.isfinite(m.vertices).all() and np.isfinite(m.face_normals).all() and (not hasattr(m.visual,'uv') or m.visual.uv is None or np.isfinite(m.visual.uv).all()) for m in s.geometry.values())
new=[m for name,m in s.geometry.items() if name.startswith('coverage_repair')];closed=[]
for m in new:
 q=m.copy();q.merge_vertices(merge_tex=True,merge_norm=True);closed.append(bool(q.is_volume))
orientation=json.loads(Path('work/orientation-area-v12/report.json').read_text());changed={r['mesh'] for r in orientation['records'] if r['repairable']};unchanged=0
for name,m in old.geometry.items():
 if name in changed:continue
 n=s.geometry[name]
 assert np.array_equal(m.vertices,n.vertices) and np.array_equal(m.faces,n.faces), name
 assert (getattr(m.visual,'uv',None) is None and getattr(n.visual,'uv',None) is None) or np.array_equal(m.visual.uv,n.visual.uv),name
 unchanged+=1
test_report=Path('work/surface-tests-v12.xml')
test_suites=ElementTree.parse(test_report).getroot().findall('.//testsuite')
assert test_suites and all(int(t.get('failures',0))+int(t.get('errors',0))==0 for t in test_suites)
passed=sum(int(t.get('tests',0))-int(t.get('skipped',0)) for t in test_suites)
report=json.loads((p/'report.json').read_text());report.update(orientationRepair=orientation,facesReoriented=orientation['facesReoriented'],validation={'finiteGeometryAndUVs':finite,'newRoofPatchesClosedAfterExport':all(closed),'newRoofPatchCount':len(new),'unrelatedMeshesExactlyPreserved':unchanged,'sourceTriangleFacesChangedOnlyByOrientation':orientation['facesReoriented'],'focusedRegressionTestsPassed':passed,'testReportSha256':hashlib.sha256(test_report.read_bytes()).hexdigest()},unresolvedAreaMeaning='Unrepaired roof-only candidate domain, including existing non-roof surfaces and evidence conflicts; this is not a measurement of visible holes.')
assert finite and all(closed)
(p/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['validation']))
