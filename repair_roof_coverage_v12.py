"""Join external area sources and apply one roof-coverage rule to all owners."""
import argparse,json,time
from pathlib import Path
import numpy as np,trimesh
from pyproj import Transformer
from shapely.geometry import shape,Polygon,Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from general_world.roof_coverage import recover_roof_coverage
from general_world.uv_transfer import apply_source_material

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--footprints',required=True);p.add_argument('--normalized-input',required=True);p.add_argument('--world',required=True);p.add_argument('--area-job',required=True);p.add_argument('--output',required=True);p.add_argument('--test-owner');a=p.parse_args();start=time.time()
path=Path(a.normalized_input);d=json.loads(path.read_text());w=json.loads(Path(a.world).read_text());frame=d['coordinateFrame'];project=Transformer.from_crs(4326,frame['projectedCrs'],always_xy=True);origin=np.array(project.transform(w['center']['lon'],w['center']['lat']));delta=np.r_[np.array(frame['originProjectedM'])-origin,frame['verticalOriginM']-w['verticalOrigin']]
cloud=np.load(path.parent/d['surfacePointsFile'])['points'][:,:3]+delta
ground=np.load(path.parent/d['groundFile'])['points'][:,:3]+delta
s=trimesh.load(a.scene,force='scene',process=False)
footprints={r['id']:shape(r['geometry']) if 'geometry' in r else unary_union([Polygon(ring) for ring in r['rings']]) for r in json.loads(Path(a.footprints).read_text())}
# The scene may retain a municipal identity after selecting a different provider
# outline. Follow the recorded source link, never substitute its older outline.
for feature in w['buildings']:
    source=feature.get('sourceUrl','')
    if source.startswith('https://www.openstreetmap.org/way/'):
        source_key='osm_'+source.rsplit('/',1)[-1]
        if source_key in footprints:footprints[feature['id']]=footprints[source_key]
owners={m.metadata.get('featureId') for m in s.geometry.values() if m.metadata.get('layer')=='roofs'}
records=[];additions=[]
job=json.loads(Path(a.area_job).read_text());boundary=Point(0,0).buffer(job['contextRadiusM'],quad_segs=96)
# Neighboring ownership records can overlap after source reconciliation. Check
# actual scene surfaces globally so such identity seams cannot create duplicate
# or hidden repair sheets.
all_upward=[];upward_heights=[]
for m in s.geometry.values():
    if m.metadata.get('layer') in ('buildings','facades','roofs'):
        for t in m.triangles[m.face_normals[:,2]>.5]:
            domain=Polygon(t[:,:2])
            if domain.area>1e-8:all_upward.append(domain);upward_heights.append(float(t[:,2].max()))
surface_index=STRtree(all_upward)
for owner in sorted(o for o in owners if o):
    if owner not in footprints or (a.test_owner and a.test_owner!=owner):continue
    meshes=[m for m in s.geometry.values() if m.metadata.get('featureId')==owner and m.metadata.get('layer')=='roofs']
    fp=footprints[owner].intersection(boundary)
    if fp.is_empty:continue
    lo=np.array(fp.bounds[:2]);hi=np.array(fp.bounds[2:]);pc=cloud[((cloud[:,:2]>=lo)&(cloud[:,:2]<=hi)).all(axis=1)];gp=ground[((ground[:,:2]>=lo-2)&(ground[:,:2]<=hi+2)).all(axis=1)]
    if not len(gp):
        records.append({'owner':owner,'status':'missing-ground-reference'});continue
    occupied=[(all_upward[i],upward_heights[i]) for i in surface_index.query(fp,predicate='intersects')]
    patches,report=recover_roof_coverage(meshes,fp,pc,float(np.median(gp[:,2]))+3.,occupied_surfaces=occupied)
    report['owner']=owner;report['emittedPatches']=len(patches);records.append(report)
    print(json.dumps({'ownersCompleted':len(records),'patches':len(patches),'supportedAreaM2':round(report['supportedAreaM2'],2)}),flush=True)
    for patch in patches:
        parent=meshes[patch.metadata['parent_index']];patch=apply_source_material(patch,parent);patch.metadata={**parent.metadata,'evidence':'lidar-supported-coverage-repair','sourceAgeVerified':False};additions.append(patch)
for i,m in enumerate(additions):
    s.add_geometry(m,geom_name='coverage_repair_'+str(i),node_name='coverage_repair_'+str(i),transform=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],float))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s.export(out/'candidate.glb')
report={'allAreaOwnersExamined':not bool(a.test_owner),'ownersExamined':len(records),'ownersChanged':sum(r.get('supportedAreaM2',0)>0 for r in records),'patchesAdded':len(additions),'supportedAreaM2':sum(r.get('supportedAreaM2',0) for r in records),'unresolvedAreaM2':sum(r.get('unresolvedAreaM2',0) for r in records),'existingGeometryPreserved':True,'facadesReconstructed':False,'sourceAgeVerified':False,'records':records,'elapsedSeconds':time.time()-start}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))
