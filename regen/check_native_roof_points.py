"""Black-box native raw-inlier guard for external roof candidates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,trimesh
from shapely import contains_xy,points as shp_points,buffer as shp_buffer
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from pipeline import roof_shell,spatial_rules
from general_world.roof_planes import RoofPolicy
from regen.rebuild_roof_support_review import contexts,parameters


def heights(mesh,xy,expected):
 tri=np.asarray(mesh.triangles);aa,bb,cc=tri[:,0],tri[:,1],tri[:,2];den=(bb[:,1]-cc[:,1])*(aa[:,0]-cc[:,0])+(cc[:,0]-bb[:,0])*(aa[:,1]-cc[:,1]);valid=abs(den)>1e-12;tri=tri[valid];den=den[valid];polys=[Polygon(q[:,:2]) for q in tri];tree=STRtree(polys);tolerance=max(float(np.spacing(np.float32(abs(xy).max())))*2,1e-7);pairs=tree.query(shp_buffer(shp_points(xy),tolerance),predicate='intersects');out=np.full(len(xy),np.nan);best=np.full(len(xy),np.inf)
 for pidx,tidx in pairs.T:
  a,b,c=tri[tidx];x,y=xy[pidx];u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den[tidx];v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den[tidx];z=u*a[2]+v*b[2]+(1-u-v)*c[2];distance=abs(z-expected[pidx])
  if distance<best[pidx]:best[pidx]=distance;out[pidx]=z
 return out,tolerance


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('build','profiles','observations','candidates','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);data=json.load(open(a.candidates/'audit.json'));eligible=[r for r in data['records'] if r['after'].get('appliedBoundaryVertices') and r['after'].get('transition',{}).get('status')=='provisional-bounded-native-supported-transition'];ids={r['building'] for r in eligible};masses=[m for m in json.load(open(a.build/'massing-dump.json')) if m['id'] in ids];saved=np.load(a.build/'rooftop-structure-returns.npz');ctx=contexts(a.build.parent,masses,saved);profiles=json.load(open(a.profiles))['profiles'];obs=json.load(open(a.observations));obs=obs.get('observations',obs);audits={r['buildingId']:r for r in json.load(open(a.build/'roof-envelope-audit.json'))};reports=[];band=RoofPolicy().residual_m
 for mass in masses:
  identity=mass['id'];foot=Polygon(mass['outline'][0]);q=ctx[identity];raw=q[contains_xy(foot,q[:,0],q[:,1])&(q[:,2]>mass['z0']+3)];x0,y0,x1,y1=foot.bounds;nx=max(2,int(np.ceil((x1-x0)/1.5)));ny=max(2,int(np.ceil((y1-y0)/1.5)));inner=raw[contains_xy(foot.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,x0,y1,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,x0,y1,1.5,nx,ny);boundary=raw[keep];constraints,host=parameters(obs,profiles,identity,mass['z0']);capture={};fit_original=roof_shell._fit_structural_regions;harmonic_original=roof_shell._harmonic_structural_heights
  def fit(g,points,*args,**kwargs):capture['points']=points.copy();return fit_original(g,points,*args,**kwargs)
  def harmonic(xy,faces,regions,*args,**kwargs):capture['regions']=regions;return harmonic_original(xy,faces,regions,*args,**kwargs)
  roof_shell._fit_structural_regions=fit;roof_shell._harmonic_structural_heights=harmonic
  try:native=roof_shell.build(foot,inner,audits[identity]['baseRoofZ'],boundary_points=boundary,eave_constraints=constraints,host_spec=host)
  finally:roof_shell._fit_structural_regions=fit_original;roof_shell._harmonic_structural_heights=harmonic_original
  after_path=a.candidates/f'{identity}-after.ply';before_path=a.candidates/f'{identity}-before.ply'
  if not after_path.exists():reports.append({'building':identity,'status':'unchanged-no-export'});continue
  after=trimesh.load(after_path,process=False);before=trimesh.load(before_path,process=False);regions=[];point_sets=[];planes=[]
  for r in capture.get('regions',[]):
   source=capture['points'];expected=np.c_[source[:,:2],np.ones(len(source))]@r['plane'];keep=contains_xy(r['geometry'].buffer(1e-7),source[:,0],source[:,1])&(abs(expected-source[:,2])<=band);inliers=source[keep]
   if not len(inliers):continue
   h0,tol=heights(before,inliers[:,:2],expected[keep]);h1,_=heights(after,inliers[:,:2],expected[keep]);hn,_=heights(native['surface'],inliers[:,:2],expected[keep]);valid=np.isfinite(h0)&np.isfinite(h1)&np.isfinite(hn);r0=abs(h0[valid]-inliers[valid,2]);r1=abs(h1[valid]-inliers[valid,2]);rh=abs(h1[valid]-hn[valid]);regions.append({'plane':r['plane'].tolist(),'rawInliers':len(inliers),'sampledRawInliers':int(valid.sum()),'missingProjectionSamples':int((~valid).sum()),'nativeFitResidualBandM':band,'pixelFreeGuard':True,'float32XYToleranceM':tol,'beforeResidualP95M':float(np.percentile(r0,95)) if len(r0) else None,'afterResidualP95M':float(np.percentile(r1,95)) if len(r1) else None,'maximumNativeSurfaceHeightChangeM':float(rh.max()) if len(rh) else None,'pointsNewlyOutsideExistingInlierBand':int(((r1>band+1e-5)&(r0<=band+1e-5)).sum()),'pointsWithNativeSurfaceChangeOver1cm':int((rh>.01).sum())});point_sets.append(inliers);planes.append(np.repeat(np.asarray(r['plane'])[None,:],len(inliers),axis=0))
  status='passed-no-new-native-inlier-violations' if regions and not any(r['missingProjectionSamples'] or r['pointsNewlyOutsideExistingInlierBand'] for r in regions) else 'withhold-native-inlier-guard-failed';report={'building':identity,'status':status,'regions':regions,'nativeRawSourcePointsSha256':hashlib.sha256(np.vstack(point_sets).astype('<f8').tobytes()).hexdigest() if point_sets else None,'sourceReadingHashes':[{ 'profileId':p['id'],'source':p.get('readingProvenance',{}).get('sourceReading')} for p in profiles if (p.get('match')or{}).get('partId')==identity]};reports.append(report);np.savez_compressed(a.output/f'{identity}-native-inliers.npz',points=np.vstack(point_sets) if point_sets else np.empty((0,3)),expectedPlanes=np.vstack(planes) if planes else np.empty((0,3)));(a.output/'audit.json').write_text(json.dumps({'status':'black-box-actual-native-raw-inlier-guard','productionChanges':0,'nativeResidualBandFromPolicyM':band,'records':reports,'qualification':'Actual structural-fit input returns inside accepted direct/validated plane regions; LAS semantic roof/tree class is not inferred. Source-camera/annotation uncertainty remains separate.'},indent=2));print(json.dumps({'building':identity,'status':status,'rawInliers':sum(r['rawInliers'] for r in regions),'newOutOfBand':sum(r['pointsNewlyOutsideExistingInlierBand'] for r in regions),'maximumNativeHeightChangeM':max((r['maximumNativeSurfaceHeightChangeM'] or 0) for r in regions)}),flush=True)
if __name__=='__main__':main()
