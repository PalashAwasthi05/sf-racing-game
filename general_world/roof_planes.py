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
from .geometry import extruded, union_solids, polygons

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
    minimum_interior_explained: float = .90   # share of interior cells under a patch its plane must explain
    equipment_m: float = 1.5                  # a return this far above a plane can be something standing on the roof
    equipment_share: float = .25              # ... while such returns cover at most this share of the patch

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

def _largest_region(indices,keys,shape):
    """Indices of the largest 8-connected group of cells among `indices`."""
    from scipy.ndimage import label
    mask=np.zeros(shape,bool);mask[keys[indices,1],keys[indices,0]]=True
    connected,count=label(mask,structure=np.ones((3,3)))
    if count<=1:return indices
    sizes=np.bincount(connected.ravel())[1:];component=int(np.argmax(sizes))+1
    return indices[connected[keys[indices,1],keys[indices,0]]==component]

def _candidates(points, policy):
    # Stable spatial ordering makes labels, input order and source IDs irrelevant.
    remaining=np.arange(len(points)); result=[]; rng=np.random.default_rng(812)
    design=np.c_[points[:,:2],np.ones(len(points))]
    minimum=max(8,int(len(points)*policy.minimum_plane_fraction))
    # A roof surface is one connected region. A candidate is scored by the largest connected group of
    # cells it explains, not by every cell within tolerance anywhere: a tilted plane through a staircase
    # of flat tiers clips a band from each tier and would outscore any one tier on a plain count.
    origin=points[:,:2].min(axis=0);keys=np.floor((points[:,:2]-origin)/policy.cell_m+1e-9).astype(int);shape=tuple(keys.max(axis=0)[::-1]+1)
    for _ in range(policy.maximum_planes):
        if len(remaining)<minimum:break
        best=None; best_count=0
        for _ in range(policy.ransac_attempts):
            inds=rng.choice(remaining,3,replace=False); c=_plane(points[inds])
            if c is None or np.linalg.norm(c[:2])>3.7:continue
            inliers=remaining[np.abs(design[remaining]@c-points[remaining,2])<=policy.residual_m]
            if len(inliers)<=best_count:continue
            region=_largest_region(inliers,keys,shape)
            if len(region)>best_count:best=region;best_count=len(region)
        if best is None or best_count<minimum:break
        c=_plane(points[best]); inliers=remaining[np.abs(design[remaining]@c-points[remaining,2])<=policy.residual_m]
        region=_largest_region(inliers,keys,shape);c=_plane(points[region]);result.append(c);remaining=np.setdiff1d(remaining,region)
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
    from .geometry import polygons
    boundary=np.concatenate([np.asarray(p.exterior.coords) for p in polygons(footprint)])
    if bottom>=float(evaluate(model,boundary).min())-.01:raise ValueError('Roof base intersects the roof envelope')
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

def fit_patches(footprint,points,policy=None):
    """Spatially validate local planes when one envelope cannot explain a roof.

    Unassigned cells stay unresolved. No nearest filling or guessed topological
    connection is made across a channel, courtyard, equipment or missing data.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import label
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    policy=policy or RoofPolicy();samples,coverage=sample_cells(footprint,points,policy)
    if len(samples)<policy.minimum_cells:return [],{'status':'insufficient-samples','acceptedPatchAreaM2':0}
    origin=np.array(footprint.bounds[:2]);keys=np.floor((samples[:,:2]-origin)/policy.cell_m).astype(int)
    held=(keys[:,0]+2*keys[:,1])%4==0;train=samples[~held];test=samples[held]
    if len(test)<6:return [],{'status':'insufficient-holdout','acceptedPatchAreaM2':0}
    planes=_candidates(train,policy)
    if not planes:return [],{'status':'no-stable-planes','acceptedPatchAreaM2':0}
    coefficients=np.asarray(planes);residual=np.abs(np.c_[train[:,:2],np.ones(len(train))]@coefficients.T-train[:,2,None])
    labels=residual.argmin(axis=1);labels[residual.min(axis=1)>policy.residual_m]=-1
    kd=cKDTree(train[:,:2]);dist,nearest=kd.query(test[:,:2],k=min(4,len(train)))
    testlabels=labels[nearest[:,0]]
    # Validate only spatially interior held-out cells: mixed neighbours denote a
    # breakline, and must not be treated as evidence for either adjacent plane.
    interior=(labels[nearest]==testlabels[:,None]).all(axis=1)&(dist[:,-1]<policy.cell_m*3)&(testlabels>=0)
    assignments=np.full(len(samples),-1);assignments[~held]=labels
    error=np.abs(np.c_[test[:,:2],np.ones(len(test))]@coefficients.T-test[:,2,None])
    tests=np.where(held)[0]
    assignments[tests[interior]]=testlabels[interior]
    patches=[];reports=[];accepted=[]
    for index,c in enumerate(planes):
        validtest=interior&(testlabels==index)
        if validtest.sum()<4:continue
        errs=error[validtest,index];p95=float(np.quantile(errs,.95))
        if p95>policy.maximum_p95_error_m:continue
        accepted.append((c,p95,int(validtest.sum())))
    if not accepted:return [],{'status':'no-validated-patches','observedCoverage':coverage,'patches':[],'acceptedPatchAreaM2':0,'unobservedCellsFilled':0}
    # Geometry: every measured cell joins the validated plane that explains it, held-out cells included.
    # The hold-out is a validation device, not a hole in the roof: leaving those cells out cut a lattice
    # of notches along every boundary, and the notch walls stood on the roof as a comb of fins.
    coefficients=np.asarray([c for c,_,_ in accepted]);residual=np.abs(np.c_[samples[:,:2],np.ones(len(samples))]@coefficients.T-samples[:,2,None])
    pad=3;nx,ny=(np.ceil((np.asarray(footprint.bounds[2:])-origin)/policy.cell_m).astype(int)+1+2*pad)
    nx=max(nx,keys[:,0].max()+1+2*pad);ny=max(ny,keys[:,1].max()+1+2*pad);grid_origin=origin-pad*policy.cell_m
    grid=np.full((ny,nx),-1,int);best=residual.argmin(axis=1);best[residual.min(axis=1)>policy.residual_m]=-1
    grid[keys[:,1]+pad,keys[:,0]+pad]=best;measured=np.zeros((ny,nx),bool);measured[keys[:,1]+pad,keys[:,0]+pad]=True
    for k in range(len(accepted)):
        connected,count=label(grid==k)
        for component in range(1,count+1):
            if (connected==component).sum()<9:grid[connected==component]=-1
    cellz=np.full((ny,nx),np.nan);cellz[keys[:,1]+pad,keys[:,0]+pad]=samples[:,2]
    joined=_join_boundary_cells(grid,footprint,grid_origin,policy.cell_m,cellz=cellz,planes=coefficients,maximum_contradiction_m=1.0)
    _regularise(grid,grid_origin,policy.cell_m,cellz=cellz,planes=coefficients,maximum_contradiction_m=1.0)
    regions=_partition(grid,grid_origin,policy.cell_m,tolerance=1.1*policy.cell_m)
    for k,(c,p95,held_count) in enumerate(accepted):
        g=regions.get(k)
        if g is None or g.is_empty:continue
        g=g.intersection(footprint)
        if g.is_empty:continue
        model={'planes':[np.asarray(c).tolist()],'operation':'min'}
        # The plane must explain the roof inside the outline it ends up with, not only the cells it chose.
        # Cells on the outline may straddle a step or carry a parapet; interior cells (a cell and a half
        # from the outline) may not disagree beyond an allowance for rooftop equipment. A plane drawn
        # through a staircase of tiers, or through two surfaces, fails here even when its own cells fit.
        under=contains_xy(g,samples[:,0],samples[:,1]);interior=contains_xy(g.buffer(-1.5*policy.cell_m),samples[:,0],samples[:,1])
        fits=on_plane=residual[:,k]<=policy.maximum_p95_error_m
        # A skylight, chimney or vent stands on the roof: cells up to equipment_m above the plane, together covering at
        # most equipment_share of the patch, are not evidence against it (one on a gable half left 4 of its 14 interior
        # cells unexplained and the roof was built flat). Cells below the plane, a storey above it, or a raised area
        # larger than equipment still are: a plane through a staircase of tiers has cells on both sides of it.
        above=samples[:,2]-(np.c_[samples[:,:2],np.ones(len(samples))]@np.asarray(c))
        on_top=~fits&(above>0)&(above<=policy.equipment_m)
        if under.any() and float(np.mean(on_top[under]))<=policy.equipment_share:fits=fits|on_top
        explained=float(np.mean(fits[under])) if under.any() else 0.;explained_interior=float(np.mean(fits[interior])) if interior.sum()>=10 else explained
        report={'areaM2':g.area,'heldOutSamples':held_count,'heldOutP95M':p95,'sloped':bool(np.linalg.norm(c[:2])>=.015),
                'outlineVertices':int(sum(len(q.exterior.coords)-1 for q in polygons(g))),'cellsUnderOutline':int(under.sum()),'explainedUnderOutline':explained,
                'interiorCells':int(interior.sum()),'explainedInterior':explained_interior}
        if explained_interior<policy.minimum_interior_explained:
            reports.append({**report,'rejected':'interior-cells-not-explained'});continue
        reports.append(report)
        # Cells under the outline the plane does not explain carry something the returns did measure (a
        # rooftop structure, a parapet): the solid passes under them, but nothing standing there is trimmed.
        unexplained=unary_union([box(*(origin+keys[i]*policy.cell_m),*(origin+(keys[i]+1)*policy.cell_m)) for i in np.where(under&~on_plane)[0]]) if (under&~on_plane).any() else None
        patches.append({'geometry':g,'model':model,'heldOutP95M':p95,'explainedUnderOutline':explained,'explainedInterior':explained_interior,'unexplained':unexplained})
    return patches,{'status':'spatially-validated-patches','observedCoverage':coverage,'patches':reports,
                    'acceptedPatchAreaM2':sum(p['geometry'].area for p in patches),'unobservedCellsFilled':int((joined&~measured).sum()),'edgeCellsJoined':int(joined.sum())}

def _join_boundary_cells(grid,footprint,origin,cell,cellz=None,planes=None,maximum_contradiction_m=1.0):
    """Unclaimed cells on a boundary join the plane of their neighbours; interior unknowns stay unknown.

    Three kinds of boundary cell, none of them a claim about an unmeasured roof:
    * a cell whose claimed neighbours belong to two different planes lies on the step or ridge between
      them (its returns are a mixture of both surfaces, so neither plane explained it); the line between
      the planes runs through it, and it goes to the side with more neighbours;
    * a cell within one cell of the footprint boundary is cut by the wall line, its returns few and mixed
      with the parapet; the surface measured beside it continues to the wall, which clips it;
    * a cell outside the footprint is filled so that the clip falls on the wall line and not on the
      raster's staircase; it is removed by the clip and carries no evidence.
    A channel, courtyard or structure the returns did not explain has neighbours of one plane at most on
    each side and stays unresolved. An edge cell measured far from the plane (more than a parapet's height,
    `maximum_contradiction_m`) is a taller neighbouring block's roof or wall, not this roof's edge, and is
    never joined: the patch must not extend under another block and trim its walls. Returns the mask of
    joined cells inside the footprint.
    """
    ny,nx=grid.shape;joined=np.zeros((ny,nx),bool)
    if footprint.is_empty:return joined
    cy,cx=np.mgrid[0:ny,0:nx];centers=np.c_[origin[0]+(cx.ravel()+.5)*cell,origin[1]+(cy.ravel()+.5)*cell]
    inside=contains_xy(footprint,centers[:,0],centers[:,1]).reshape(ny,nx)
    core=contains_xy(footprint.buffer(-cell),centers[:,0],centers[:,1]).reshape(ny,nx)
    near=contains_xy(footprint.buffer(2.5*cell),centers[:,0],centers[:,1]).reshape(ny,nx)
    def consistent(y,x,k):
        if cellz is None or planes is None or not np.isfinite(cellz[y,x]):return True
        a,b,c=planes[k];return abs(a*(origin[0]+(x+.5)*cell)+b*(origin[1]+(y+.5)*cell)+c-cellz[y,x])<=maximum_contradiction_m
    def sweep(candidate,passes,require_two_planes=False,check_height=False):
        for _ in range(passes):
            updates=[]
            for y,x in np.argwhere(candidate&(grid<0)):
                window=grid[max(0,y-1):y+2,max(0,x-1):x+2];values=window[window>=0]
                if not len(values) or (require_two_planes and len(np.unique(values))<2):continue
                k=int(np.bincount(values).argmax())
                if check_height and not consistent(y,x,k):continue
                updates.append((y,x,k))
            if not updates:break
            for y,x,k in updates:grid[y,x]=k;joined[y,x]|=inside[y,x]
    sweep(inside&core,2,require_two_planes=True)
    sweep(inside&~core,2,check_height=True)
    sweep(near&~inside,3,check_height=True)
    return joined

def _regularise(grid,origin,cell,cellz=None,planes=None,maximum_contradiction_m=1.0):
    """Roof regions are at least three cells wide everywhere; thinner teeth and notches are raster noise.

    Where a slope meets the flat roof it rises from, cells within tolerance of both surfaces are assigned
    by whichever residual happens to be smaller, and the eave line comes out as a saw of teeth a cell or
    two wide. A binary opening removes such teeth and a closing fills such notches (3 x 3 structure, the
    same 3 m minimum width the generator applies to roof tiers). A notch cell joins only when unclaimed and
    not measured more than `maximum_contradiction_m` from the plane; another region's cells are never taken.
    """
    from scipy.ndimage import binary_opening,binary_closing
    structure=np.ones((3,3),bool);labels=[int(k) for k in np.unique(grid[grid>=0])]
    labels.sort(key=lambda k:-(grid==k).sum())
    for k in labels:
        mask=grid==k;opened=binary_opening(mask,structure);closed=binary_closing(opened,structure)
        grid[mask&~opened]=-1
        for y,x in np.argwhere(closed&~opened&(grid<0)):
            if cellz is None or planes is None or not np.isfinite(cellz[y,x]):grid[y,x]=k;continue
            a,b,c=planes[k]
            if abs(a*(origin[0]+(x+.5)*cell)+b*(origin[1]+(y+.5)*cell)+c-cellz[y,x])<=maximum_contradiction_m:grid[y,x]=k

def _partition(grid,origin,cell,tolerance):
    """Polygons of each label with their shared boundaries simplified consistently.

    A boundary read from a raster is a staircase of cell edges; a building edge or the step between two
    roof sections is straight wherever the raster shows it straight to within its resolution. Boundary
    chains between junctions are simplified with their ends fixed and the faces rebuilt, so neighbouring
    regions keep one common boundary (no slivers, no overlaps). The tolerance is 1.1 cells: a staircase's
    corners lie up to a cell from the chord between its ends, while a real corner in the boundary deviates
    by the length of its legs and survives. A region whose simplified outline strays further than the
    tolerance from the raster falls back to its raster outline.
    """
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    from shapely.ops import linemerge,polygonize
    ny,nx=grid.shape;transform=from_origin(origin[0],origin[1]+ny*cell,cell,cell);raw={}
    for k in np.unique(grid[grid>=0]):
        mask=(grid==k)[::-1]
        geoms=[shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=transform) if v]
        if geoms:raw[int(k)]=unary_union(geoms)
    if not raw or tolerance<=0:return raw
    lines=unary_union([p.boundary for p in raw.values()])
    merged=linemerge(lines) if lines.geom_type=='MultiLineString' else lines
    chains=list(getattr(merged,'geoms',[merged]))
    noded=unary_union([c.simplify(tolerance,preserve_topology=True) for c in chains])
    result={}
    for face in polygonize(noded):
        point=face.representative_point();owner=next((k for k,p in raw.items() if p.contains(point)),None)
        if owner is None:
            overlaps={k:face.intersection(p).area for k,p in raw.items()};owner=max(overlaps,key=overlaps.get) if max(overlaps.values())>0 else None
        if owner is None:continue
        result[owner]=unary_union([result[owner],face]) if owner in result else face
    for k,p in raw.items():
        if k not in result or result[k].symmetric_difference(p).area>1.2*tolerance*p.length:result[k]=p
    return result
