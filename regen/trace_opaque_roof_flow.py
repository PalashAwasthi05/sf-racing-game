"""Read-only trace from source roof observations to emitted opaque wall faces."""
import argparse,json,types
from pathlib import Path
import numpy as np
import trimesh
from shapely import contains_xy
from shapely.geometry import Polygon,Point,box
from pipeline import roof_shell,spatial_rules,roof_assemblies
from regen.rebuild_roof_support_review import parameters
from regen.check_roof_sky_projection import sha


def trace(a):
    masses=json.loads((a.build/'massing-dump.json').read_text());audits=json.loads((a.build/'roof-envelope-audit.json').read_text());profiles=json.loads(a.profiles.read_text())['profiles'];observations=json.loads(a.observations.read_text());observations=observations.get('observations',observations);jobs=json.loads(a.jobs.read_text());scene=trimesh.load(a.build/'king-street-pilot.glb',force='scene');clouds=np.load(a.build/'rooftop-structure-returns.npz');facades=json.loads((a.build/'king-street-evidence.json').read_text())['facadeWalls']
    results=[]
    for row in a.rows:
        job=jobs[row];identity=job['building'];mass=next(m for m in masses if m['id']==identity);audit=next(r for r in audits if r['buildingId']==identity);obs=[o for o in observations if o['side'].startswith(identity+'__')];joined=[p for p in profiles if (p.get('match') or {}).get('partId')==identity];footprint=Polygon(mass['outline'][0],mass['outline'][1:]);raw=clouds[identity+'__context'];raw=raw[contains_xy(footprint,raw[:,0],raw[:,1])&(raw[:,2]>mass['z0']+3)]
        x0,y0,x1,y1=footprint.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)));inner=raw[contains_xy(footprint.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];constraints,host=parameters(observations,profiles,identity,mass['z0'])
        capture={};module=types.ModuleType('pipeline._roof_flow_readonly');module.__package__='pipeline';exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),module.__dict__);original=module._harmonic_structural_heights
        def inspect(xy,faces,regions,*args,**kwargs):
            result=original(xy,faces,regions,*args,**kwargs);capture.update(xy=xy.copy(),known=result[1].copy(),unknown=result[2].copy(),regions=regions);return result
        module._harmonic_structural_heights=inspect
        shell=module.build(footprint,inner,audit['baseRoofZ'],boundary_points=boundary,eave_constraints=constraints,host_spec=host)
        source_records=[]
        roofs=trimesh.util.concatenate([g for k,g in scene.geometry.items() if k.startswith('roofs__'+identity+'___')])
        for p in joined:
            n=np.asarray(p['normal']);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]]);left,right=p['frameU'];plane=p['planeV'];source_top=(p.get('reading') or {}).get('wallTopM');line_results=[]
            if capture:
                xy=capture['xy'];local_x=xy@t-left;on=(abs(xy@n-plane)<.08)&(local_x>=-.1)&(local_x<=right-left+.1)
                line_results={'sampledBoundaryVertices':int(on.sum()),'knownPlaneOrEaveAnchors':int(capture['known'][on].sum()),'unknownHarmonicVertices':int(capture['unknown'][on].sum())}
            emitted=[]
            if source_top is not None:
                for name,mesh in scene.geometry.items():
                    if not name.startswith(('buildings__'+identity+'__','facades__'+identity+'__')):continue
                    tri=np.asarray(mesh.triangles);residual=abs(tri[:,:,:2]@n-plane).max(1);normals=mesh.face_normals
                    select=(residual<.08)&(abs(normals[:,:2]@n)>.95)&(tri[:,:,2].max(1)>mass['z0']+source_top)
                    area=0.;top=0.;count=0
                    for q in tri[select]:
                        local=np.c_[q[:,:2]@t-left,q[:,2]-mass['z0']];poly=Polygon(local)
                        if not poly.is_valid:continue
                        piece=poly.intersection(box(0,source_top,right-left,100))
                        if piece.area>1e-6:area+=piece.area;top=max(top,float(local[:,1].max()));count+=1
                    if count:emitted.append({'geometry':name,'frontAreaAboveSourceWallTopM2':area,'maximumHeightAboveBuildingGroundM':top,'faces':count})
            component=[]
            for observation in obs:
                if observation['side']!=p['id'].removeprefix('regen-'):continue
                for feature in observation.get('features',[]):
                    if feature.get('kind')!='dormer':continue
                    anchor,_,_=roof_assemblies._xy(p,left+(feature['x0M']+feature['x1M'])/2,plane);z_target=mass['z0']+feature['bottomM'];tri=roofs.triangles;aa,bb,cc=tri[:,0],tri[:,1],tri[:,2];den=(bb[:,1]-cc[:,1])*(aa[:,0]-cc[:,0])+(cc[:,0]-bb[:,0])*(aa[:,1]-cc[:,1]);valid=abs(den)>1e-10;tri=tri[valid];aa,bb,cc,den=aa[valid],bb[valid],cc[valid],den[valid];samples=[]
                    for inset in roof_assemblies._support_insets(anchor,n,footprint):
                        point=anchor-n*inset;x,y=point;u=((bb[:,1]-cc[:,1])*(x-cc[:,0])+(cc[:,0]-bb[:,0])*(y-cc[:,1]))/den;v=((cc[:,1]-aa[:,1])*(x-cc[:,0])+(aa[:,0]-cc[:,0])*(y-cc[:,1]))/den;inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
                        if inside.any():
                            z=u[inside]*tri[inside,0,2]+v[inside]*tri[inside,1,2]+(1-u[inside]-v[inside])*tri[inside,2,2];samples.append({'insetM':float(inset),'surfaceHeightM':float(z.max()-mass['z0']),'aboveFeatureBottomM':float(z.max()-z_target)})
                    component.append({'boundsM':{k:feature.get(k) for k in ('x0M','x1M','bottomM','topM')},'hasSourceFront':roof_assemblies._has_source_front(feature),'boundsEvidenceStatus':(feature.get('boundsEvidence') or {}).get('status'),'supportRaySamples':samples})
            source_records.append({'profileId':p['id'],'sourceFrameHeightM':p['heightRefM'],'sourceWallTopM':source_top,'legacyRoofs':p.get('roofs'),'readingProvenance':p.get('readingProvenance'),'frontBoundaryOwnership':line_results,'emittedOpaqueFrontAboveSourceWallTop':emitted,'components':component})
        original=audit['roofShell']['structuralPlaneSurface'].get('planes',[]);reproduced=shell['audit']['structuralPlaneSurface'].get('planes',[]);corners=np.array([[x,y,1] for x in (x0,x1) for y in (y0,y1)]);errors=[min(float(abs(corners@(np.asarray(p)-q)).max()) for q in reproduced) for p in original]
        results.append({'row':row,'building':identity,'massing':mass,'roofObservationSummary':[{k:o.get(k) for k in ('side','confidence','hostRoofKind','eaveM','eaveConfidence','anchorUncertaintyM','hostRidgeM','visibleRoofDepthM','roofPartUnresolved')} for o in obs],'sourceProfiles':source_records,'hostAudit':audit['roofShell']['observedHostForm'],'componentAudit':audit['observation']['features'],'boundaryAudit':{k:audit['roofShell'].get(k) for k in ('boundarySampleHeightRangeM','boundaryConstraintsAccepted','boundaryConstraintVerticesApplied','uncertainBoundaryConstraintCount','uncertainBoundaryConstraintsAppliedAsHardAnchors')},'localFrontPlaneRecovery':audit['roofShell']['structuralPlaneSurface'].get('localFrontPlaneRecovery'),'facadeAudit':[{k:r.get(k) for k in ('grammarId','heightM','wallTopM','wallLevelM','frameHeightM','lengthM')} for r in facades if r['buildingId']==identity and r.get('grammarId')],'reproductionMaximumPlaneErrorAcrossFootprintM':max(errors,default=None)})
        print(json.dumps({'row':row,'reproductionPlaneErrorM':max(errors,default=None),'profiles':len(source_records)}),flush=True)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'audit.json').write_text(json.dumps({'status':'read-only-dataflow-diagnosis','inputs':{k:{'path':str(getattr(a,k)),'sha256':sha(getattr(a,k))} for k in ('profiles','observations','jobs')},'actualExportPath':str(a.build/'king-street-pilot.glb'),'actualExportSha256':sha(a.build/'king-street-pilot.glb'),'roofRuleSha256':sha(roof_shell.__file__),'rows':results,'productionChanges':0},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','profiles','observations','jobs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--rows',type=int,nargs='+',required=True);trace(p.parse_args())
