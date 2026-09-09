"""Evidence-tested continuous roof planes, without roof-style or location lookups.

A plane envelope is accepted only when it explains held-out spatial samples.
Complex stepped, curved, occluded or sparse roofs remain unresolved by this fitter.
"""
from dataclasses import dataclass
import numpy as np
import trimesh
from shapely import contains_xy
from shapely.geometry import box
from shapely.ops import unary_union
from .geometry import extruded, union_solids

@dataclass(frozen=True)
class RoofPolicy:
    cell_m: float = 1.0
    minimum_points_per_cell: int = 3
    minimum_cells: int = 30
    minimum_coverage: float = .80
    residual_m: float = .22
    maximum_p95_error_m: float = .30
    minimum_explained_fraction: float = .95
    minimum_plane_fraction: float = .08
    maximum_planes: int = 6
    ransac_attempts: int = 240

def sample_cells(footprint, points, policy):
    p=np.asarray(points,float)[:,:3]
    p=p[np.isfinite(p).all(axis=1)]
    p=p[contains_xy(footprint,p[:,0],p[:,1])]
    origin=np.array(footprint.bounds[:2]); cells=np.floor((p[:,:2]-origin)/policy.cell_m).astype(int)
    groups={}
    for index,key in enumerate(map(tuple,cells)): groups.setdefault(key,[]).append(index)
    samples=[]; supported=[]
    for key,indices in sorted(groups.items()):
        if len(indices)<policy.minimum_points_per_cell: continue
        q=p[indices]; q=q[q[:,2]>=np.quantile(q[:,2],.5)]
        samples.append(np.median(q,axis=0))
        lo=origin+np.array(key)*policy.cell_m
        supported.append(box(*lo,*(lo+policy.cell_m)))
    coverage=unary_union(supported).intersection(footprint).area/max(footprint.area,1e-9) if supported else 0
    return np.asarray(samples).reshape(-1,3),float(coverage)

def _plane(points):
    a=np.c_[points[:,:2],np.ones(len(points))]
    if np.linalg.matrix_rank(a)<3:return None
    return np.linalg.lstsq(a,points[:,2],rcond=None)[0]

def _candidates(points, policy):
    # Stable spatial ordering makes labels, input order and source IDs irrelevant.
    remaining=np.arange(len(points)); result=[]; rng=np.random.default_rng(812)
    design=np.c_[points[:,:2],np.ones(len(points))]
    minimum=max(8,int(len(points)*policy.minimum_plane_fraction))
    for _ in range(policy.maximum_planes):
        if len(remaining)<minimum:break
        best=None; best_count=0
        for _ in range(policy.ransac_attempts):
            inds=rng.choice(remaining,3,replace=False); c=_plane(points[inds])
            if c is None or np.linalg.norm(c[:2])>3.7:continue
            inliers=remaining[np.abs(design[remaining]@c-points[remaining,2])<=policy.residual_m]
            if len(inliers)>best_count:best=inliers;best_count=len(inliers)
        if best is None or best_count<minimum:break
        c=_plane(points[best]); inliers=remaining[np.abs(design[remaining]@c-points[remaining,2])<=policy.residual_m]
        c=_plane(points[inliers]);result.append(c);remaining=np.setdiff1d(remaining,inliers)
    return result

def evaluate(model,xy):
    xy=np.asarray(xy); heights=np.c_[xy.reshape(-1,2),np.ones(xy.size//2)]@np.asarray(model['planes']).T
    z=heights.min(axis=1) if model['operation']=='min' else heights.max(axis=1)
    return z.reshape(xy.shape[:-1])

def fit_roof(footprint, points, policy=None):
    policy=policy or RoofPolicy(); samples,coverage=sample_cells(footprint,points,policy)
    report={'accepted':False,'coverage':coverage,'sampleCells':len(samples),'status':'insufficient-spatial-support'}
    if len(samples)<policy.minimum_cells or coverage<policy.minimum_coverage:return None,report
    samples=samples[np.lexsort((samples[:,1],samples[:,0]))]
    # A spatial checkerboard holds out cells rather than duplicate points.
    keys=np.floor((samples[:,:2]-np.array(footprint.bounds[:2]))/policy.cell_m).astype(int)
    held=((keys[:,0]+2*keys[:,1])%4)==0
    if held.sum()<6 or (~held).sum()<12:return None,report
    train=samples[~held]; test=samples[held]; planes=_candidates(train,policy)
    options=[]
    for count in range(1,len(planes)+1):
        for operation in ['min','max'] if count>1 else ['min']:
            model={'planes':np.asarray(planes[:count]).tolist(),'operation':operation}
            error=np.abs(evaluate(model,test[:,:2])-test[:,2]); whole=np.abs(evaluate(model,samples[:,:2])-samples[:,2])
            score={'heldOutP95M':float(np.quantile(error,.95)),'heldOutMedianM':float(np.median(error)),
                   'explainedFraction':float(np.mean(whole<=policy.maximum_p95_error_m))}
            if score['heldOutP95M']<=policy.maximum_p95_error_m and score['explainedFraction']>=policy.minimum_explained_fraction:
                options.append((count,score['heldOutP95M'],model,score))
    if not options:return None,{**report,'status':'not-explained-by-plane-envelope','candidatePlanes':len(planes)}
    _,_,model,score=min(options,key=lambda x:(x[0],x[1]))
    # Retain a simple horizontal model when slope is not measurably established.
    if len(model['planes'])==1 and np.linalg.norm(model['planes'][0][:2])<.015:
        model['planes']=[[0.,0.,float(np.median(train[:,2]))]]
        e=np.abs(evaluate(model,test[:,:2])-test[:,2])
        score['heldOutP95M']=float(np.quantile(e,.95))
        if score['heldOutP95M']>policy.maximum_p95_error_m:return None,{**report,'status':'uncertain-shallow-slope'}
    return model,{**report,**score,'accepted':True,'status':'held-out-plane-envelope','planes':len(model['planes']),
                  'sloped':any(np.linalg.norm(p[:2])>=.015 for p in model['planes']),
                  'extrapolation':'within footprint; accepted aggregate coverage, not proof of every cell'}

def roof_solid(footprint, model, bottom):
    """Exact shared ridge/eave intersections through solid half-space clipping."""
    import manifold3d
    corners=np.array([[footprint.bounds[i],footprint.bounds[j]] for i in (0,2) for j in (1,3)])
    top=float(np.max(np.c_[corners,np.ones(4)]@np.asarray(model['planes']).T))+1
    if bottom>=float(evaluate(model,corners).min())-.01:raise ValueError('Roof base intersects the roof envelope')
    def clipped(planes):
        parts=[]
        for mesh in extruded(footprint,bottom,top):
            m=manifold3d.Manifold(manifold3d.Mesh64(mesh.vertices.astype(float),mesh.faces.astype(np.uint64)))
            for a,b,c in planes:
                n=np.array([a,b,-1.]);m=m.trim_by_plane(n.tolist(),-c/np.linalg.norm(n))
            raw=m.to_mesh64();parts.append(trimesh.Trimesh(np.asarray(raw.vert_properties)[:,:3],np.asarray(raw.tri_verts),process=False))
        return union_solids(parts)
    mesh=clipped(model['planes']) if model['operation']=='min' else union_solids([clipped([p]) for p in model['planes']])
    if not mesh.is_volume:raise ValueError('Roof failed closed solid test')
    return mesh
