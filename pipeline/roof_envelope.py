"""One measured roof envelope shared by caps, massing and exterior wall tops.

Unknown samples are not evidence for roof tiers. A surface classified as sloped
must remain a surface: it must never be replaced by a zero-height tier cell.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, generic_filter
from shapely import contains_xy
from shapely.ops import unary_union
from general_world.roof_planes import fit_patches,RoofPolicy,evaluate


def prepare_grid(raw,footprint,xmin,ymax,step):
    rows,cols=np.indices(raw.shape)
    xy=np.c_[xmin+(cols.ravel()+.5)*step,ymax-(rows.ravel()+.5)*step]
    inside=contains_xy(footprint,xy[:,0],xy[:,1]).reshape(raw.shape)
    known=np.isfinite(raw)&inside
    if not known.any():return None,known,xy,{'measuredCells':0}
    # Median filtering must not borrow a neighbouring parcel or the bounding
    # rectangle outside an L-shaped building. Small noise removal only.
    measured=np.where(known,raw,np.nan)
    def median(values):
        finite=values[np.isfinite(values)]
        return np.median(finite) if len(finite) else np.nan
    filtered=generic_filter(measured,median,size=3,mode='constant',cval=np.nan)
    filtered[~inside]=np.nan
    unsupported=~np.isfinite(filtered)
    nearest=distance_transform_edt(unsupported,return_distances=False,return_indices=True)
    filled=filtered.copy();filled[unsupported]=filtered[tuple(nearest[:,unsupported])]
    # Filled values support a continuous provisional base only, not new tiers.
    return filled,known,xy,{'measuredCells':int(known.sum()),'insideCells':int(inside.sum()),'unknownInsideCells':int((inside&~known).sum())}


def base_level(rr,known,step,footprint_area,quantile=.15,band_m=1.5,minimum_share=.05,minimum_area_m2=25.,minimum_interior=.1):
    """The base roof every tier stands on: the 15th percentile of the measured roof, unless a lower roof of real extent
    lies under it. The percentile stands for 'ignore the lowest few cells', but a real lower wing holding 14-18 % of a
    roof flipped it across the gap between two levels: a small change in how the cells fall moved one building's base
    from 8 m to 16 m and stood its low wing 8 m too high. A lower level counts as a roof when a connected region of its
    cells (within band_m of it) holds at least max(minimum_area_m2, minimum_share of the footprint) and a share of them
    (minimum_interior) have measured roof on all four sides: a ring of cells straddling the walls has no interior. The
    lowest such level is the base."""
    from scipy.ndimage import label
    z=rr[known]
    if not len(z):return None
    q=float(np.quantile(z,quantile));need=max(minimum_area_m2,minimum_share*footprint_area)
    k=np.pad(known,1);interior=k[:-2,1:-1]&k[2:,1:-1]&k[1:-1,:-2]&k[1:-1,2:]&known
    def roof_at(L):
        lab,n=label(known&(np.abs(rr-L)<=band_m),structure=np.ones((3,3),bool))
        for j in range(1,n+1):
            m=lab==j
            if m.sum()*step*step>=need and interior[m].mean()>=minimum_interior:return float(np.median(rr[m]))
        return None
    for L in np.unique(np.round(z[z<q-3.]*2)/2):          # a storey or more below: nearer levels are the same roof's falls
        # a step, not a slope: between the lower level and the percentile the heights are nearly empty (a pitched or
        # sawtooth roof runs through them continuously, and its eave band is no lower roof)
        if np.mean((z>L+band_m)&(z<q-band_m))>.03:continue
        r=roof_at(L)
        if r is not None:return min(q,r)
    # The percentile itself can land in the empty gap between two levels less than a storey apart (a lower roof holding
    # 10-15 % of the cells): a base there is neither roof. A roof 36 m high over a lower strip at 33 m took a base of
    # 34.4 m, with the main roof merged down onto it, 2 m low. It moves to the level below when that is a roof of real
    # extent (the main roof then stands on it as a tier), else to the level above.
    if np.mean(np.abs(z-q)<=.5)<.03:
        below,above=z[z<q-.5],z[z>q+.5]
        if len(below):
            L=float(np.median(below[below>=below.max()-band_m]));r=roof_at(L)
            if r is not None:return r
        if len(above):return float(np.median(above[above<=above.min()+band_m]))
    return q


PITCH_SLOPE,PITCH_MIN_AREA_M2=.15,25.


def continuous_patches(footprint,points,step,min_slope=.015,min_area_m2=25.,min_rise_m=.6):
    """Tilted roof planes the fit supports. The thresholds come from the area config's LiDAR role: a drainage
    fall of a percent or two is a flat roof, and splitting it into separately tilted planes painted the roof
    in irregular patches."""
    patches,report=fit_patches(footprint,points,RoofPolicy(cell_m=step))
    accepted=[]
    for p in patches:
        if not any(np.linalg.norm(c[:2])>=min_slope for c in p['model']['planes']):continue
        # A pitched roof (15 % or steeper: a gable's planes, a hip) is a roof shape, not a drainage fall mistaken for
        # one; it keeps its planes from 25 m2. The area floor for gentle slopes flattened a small building's gable
        # (two planes of 100 m2 each fitted to 5-7 cm, 18 degrees) into a flat roof at its eaves.
        pitched=max(np.linalg.norm(c[:2]) for c in p['model']['planes'])>=PITCH_SLOPE
        for g in ([p['geometry']] if p['geometry'].geom_type=='Polygon' else list(p['geometry'].geoms)):
            if g.area<(min(min_area_m2,PITCH_MIN_AREA_M2) if pitched else min_area_m2) or g.buffer(-1.5).is_empty:continue
            xy=np.asarray(g.exterior.coords);z=evaluate(p['model'],xy)
            if z.max()-z.min()<min_rise_m:continue
            accepted.append({**p,'geometry':g})
    return join_ridges(accepted,points),report


def join_ridges(patches,points,tolerance_m=.30):
    """Adjacent pitched planes that meet in a ridge are one roof: the lower of their planes everywhere (a gable, a hip).
    Built separately, each half ended at the raster line between them, so the halves stood at different heights along
    the ridge with a wall between. A pair joins only when each plane is the lower one over its own half (a valley is
    not a ridge) and the joined roof explains the returns under both halves as well as the halves do."""
    from shapely import contains_xy
    from shapely.ops import unary_union
    pitched=[i for i,p in enumerate(patches) if len(p['model']['planes'])==1 and np.linalg.norm(np.asarray(p['model']['planes'][0])[:2])>=PITCH_SLOPE]
    group={i:{i} for i in range(len(patches))}
    pts=np.asarray(points)[:,:3]
    def own(i):
        g=patches[i]['geometry'];m=contains_xy(g,pts[:,0],pts[:,1]);return pts[m]
    def err(model,q):
        return np.quantile(np.abs(evaluate(model,q[:,:2])-q[:,2]),.8) if len(q) else 0.
    for a in pitched:
        for b in pitched:
            if b<=a or group[a] is group[b]:continue
            ga,gb=patches[a]['geometry'],patches[b]['geometry']
            if ga.buffer(.8).intersection(gb.buffer(.8)).area<1.:continue
            qa,qb=own(a),own(b)
            if len(qa)<20 or len(qb)<20:continue
            pa,pb=np.asarray(patches[a]['model']['planes'][0]),np.asarray(patches[b]['model']['planes'][0])
            za=lambda q,pl:np.c_[q[:,:2],np.ones(len(q))]@pl
            # each plane lower than the other over its own half (the ridge between them is a maximum)
            if np.quantile(za(qa,pa)-za(qa,pb),.9)>tolerance_m or np.quantile(za(qb,pb)-za(qb,pa),.9)>tolerance_m:continue
            joined={'planes':[pa.tolist(),pb.tolist()],'operation':'min'}
            if err(joined,qa)>err(patches[a]['model'],qa)+.05 or err(joined,qb)>err(patches[b]['model'],qb)+.05:continue
            merged=group[a]|group[b]
            for k in merged:group[k]=merged
    out=[];seen=set()
    for i,p in enumerate(patches):
        members=sorted(group[i])
        if id(group[i]) in seen:continue
        seen.add(id(group[i]))
        if len(members)==1:out.append(p);continue
        # the whole group, not only each adjacent pair: the lowest plane must explain every member's returns
        joined={'planes':[pl for k in members for pl in patches[k]['model']['planes']],'operation':'min'}
        if any(err(joined,own(k))>err(patches[k]['model'],own(k))+.05 for k in members):
            out.extend(patches[k] for k in members);continue
        g=unary_union([patches[k]['geometry'].buffer(.05) for k in members]).buffer(-.05)
        planes=[pl for k in members for pl in patches[k]['model']['planes']]
        unexplained=[patches[k].get('unexplained') for k in members if patches[k].get('unexplained') is not None]
        for piece in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
            out.append({**patches[members[0]],'geometry':piece,'model':{'planes':planes,'operation':'min'},'ridgeOf':len(members),
                        'unexplained':unary_union(unexplained) if unexplained else None})
    return out


def owned_samples(footprint,xy,base,bands,patches,lids=()):
    # The facade sampler expects enough samples along short wall pieces. These
    # are samples of the generated envelope, explicitly not extra observations.
    xmin,ymin,xmax,ymax=footprint.bounds
    xx,yy=np.meshgrid(np.arange(xmin+.25,xmax,.5),np.arange(ymin+.25,ymax,.5))
    xy=np.c_[xx.ravel(),yy.ravel()]
    xy=xy[contains_xy(footprint,xy[:,0],xy[:,1])]
    z=np.full(len(xy),base,dtype=float)
    for band in bands:
        mask=contains_xy(band['geometry'].buffer(.01),xy[:,0],xy[:,1]);z[mask]=base+band['top']
    for p in patches:
        mask=contains_xy(p['geometry'].buffer(.01),xy[:,0],xy[:,1]);z[mask]=evaluate(p['model'],xy[mask])
    for polygon,height in lids:
        mask=contains_xy(polygon.buffer(.01),xy[:,0],xy[:,1]);z[mask]=height
    return np.c_[xy,z]


def drape_regions(rr, known, footprint, xmin, ymax, step, minimum_area_m2=800., minimum_rise_m=6.,
                  minimum_grade=.12, maximum_roughness_m=3.0):
    """Cells on a continuous measured slope (a seating bowl, a ramp, a pitched roof no plane fit accepted), grouped
    into regions large enough to matter and rising at least `minimum_rise_m` across themselves. Such a region is one
    draped surface, never a stack of walled tiers."""
    from scipy import ndimage
    from shapely.geometry import shape
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from .spatial_rules import sloped_cells
    if rr is None: return np.zeros((0, 0), bool), None
    # The slope is read on a smoothed surface: plant, parapets and noise on a flat roof make local gradients but no
    # steady grade. A region drapes only if it is smooth (small departure from the smoothed surface), has a steady
    # grade, covers enough area and rises enough across itself.
    filled = np.where(np.isfinite(rr), rr, np.nanmedian(rr))
    smooth = ndimage.gaussian_filter(filled, 2.0)
    gy, gx = np.gradient(smooth, step); grade = np.hypot(gx, gy)
    rough = np.abs(filled - smooth)
    # A slope rises cell by cell; two flat roofs with a step between them are flat almost everywhere, and only the smoothed
    # surface shows a grade across them. The unsmoothed grade (3 x 3 median, so single noisy cells do not count) must reach
    # the minimum over most of a region, or it is a step: a flat roof ring beside a lower court was draped as one slope.
    ry, rx = np.gradient(ndimage.median_filter(filled, 3), step); raw_grade = np.hypot(rx, ry)
    mask = ((grade >= minimum_grade) & (grade <= 1.0) | sloped_cells(rr, step)) & known   # smoothed grade or a raw continuous slope
    lab, n = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * step * step < minimum_area_m2: continue
        # the rise is what the returns measure across the region: on the smoothed surface a flat roof ring beside a low
        # courtyard 'rose' 6 m across itself (the courtyard blurred into it) and the whole ring was draped as a slope
        if np.nanpercentile(filled[m], 95) - np.nanpercentile(filled[m], 5) < minimum_rise_m: continue
        if np.median(rough[m]) > maximum_roughness_m or np.median(grade[m]) < minimum_grade or np.median(raw_grade[m]) < minimum_grade: continue
        # A smoothed step between two roof levels also shows a grade. A slope spreads its raw heights evenly over its
        # range; a step clusters them in two groups with a gap. Reject regions whose largest gap between height
        # percentiles exceeds a third of their range.
        q = np.nanpercentile(filled[m], np.arange(5, 96, 3))
        if q[-1] - q[0] <= 0 or np.max(np.diff(q)) > (q[-1] - q[0]) / 3: continue
        keep |= m
    drape_regions.smooth = smooth
    if not keep.any(): return keep, None
    tr = from_origin(xmin, ymax, step, step)
    area = unary_union([shape(s) for s, v in shapes(keep.astype('uint8'), mask=keep, transform=tr) if v]).intersection(footprint)
    return keep, area


def drape_mesh(rr, keep, xmin, ymax, step, base_z):
    """The draped surface over `keep` cells (corner heights averaged from the adjoining measured cells) and the skirt
    faces that close its open edges down to the base roof. Returns (surface, skirt) as trimesh objects."""
    import trimesh
    H, W = rr.shape
    z = np.where(np.isfinite(rr), np.maximum(rr, base_z), np.nan)
    pad = np.pad(z, 1, constant_values=np.nan)
    corners = np.nanmean(np.stack([pad[:-1, :-1], pad[:-1, 1:], pad[1:, :-1], pad[1:, 1:]]), axis=0)  # (H+1, W+1)
    corners = np.where(np.isfinite(corners), corners, base_z)
    vid = lambda i, j: i * (W + 1) + j
    ii, jj = np.meshgrid(np.arange(H + 1), np.arange(W + 1), indexing='ij')
    verts = np.c_[xmin + jj.ravel() * step, ymax - ii.ravel() * step, corners.ravel()]
    faces = []
    for i, j in zip(*np.nonzero(keep)):
        a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
        faces += [[a, d, c], [a, c, b]]
    surface = trimesh.Trimesh(verts, np.array(faces), process=False)
    surface.remove_unreferenced_vertices()
    skirt_v, skirt_f = [], []
    def side(p, q):   # p,q: (i,j) corner indices along an open edge; the face stands from base_z to the surface
        (i0, j0), (i1, j1) = p, q
        x0, y0, z0 = xmin + j0 * step, ymax - i0 * step, corners[i0, j0]
        x1, y1, z1 = xmin + j1 * step, ymax - i1 * step, corners[i1, j1]
        if max(z0, z1) - base_z < .05: return
        k = len(skirt_v); skirt_v.extend([[x0, y0, base_z], [x1, y1, base_z], [x1, y1, z1], [x0, y0, z0]])
        skirt_f.extend([[k, k + 1, k + 2], [k, k + 2, k + 3], [k, k + 2, k + 1], [k, k + 3, k + 2]])  # both sides
    inside = lambda i, j: 0 <= i < H and 0 <= j < W and keep[i, j]
    for i, j in zip(*np.nonzero(keep)):
        if not inside(i - 1, j): side((i, j), (i, j + 1))
        if not inside(i + 1, j): side((i + 1, j + 1), (i + 1, j))
        if not inside(i, j - 1): side((i + 1, j), (i, j))
        if not inside(i, j + 1): side((i, j + 1), (i + 1, j + 1))
    skirt = trimesh.Trimesh(np.array(skirt_v), np.array(skirt_f), process=False) if skirt_v else None
    return surface, skirt


def cells_area(keep, footprint, xmin, ymax, step):
    """Polygon covered by raster cells `keep`, clipped to the footprint."""
    from shapely.geometry import shape
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    return unary_union([shape(s) for s, v in shapes(keep.astype('uint8'), mask=keep, transform=from_origin(xmin, ymax, step, step)) if v]).intersection(footprint)


def venue_form(rr, known, footprint, xmin, ymax, step, z0, row_rise_m=.9, minimum_field_m2=2500.):
    """Form of a stadium-scale structure from what the measurement gives reliably: the field (the large open region
    at ground level), the outer ring (a band inside the outline, at its measured height) and the seating bowl between
    them, raked in rows from the ring down to the field edge. The noisy bowl returns never drive geometry directly.
    Returns None when no open field is measured (the structure is then an ordinary building)."""
    from scipy import ndimage
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    H, W = rr.shape
    inside = rasterize([(footprint, 1)], out_shape=(H, W), transform=from_origin(xmin, ymax, step, step), fill=0).astype(bool)
    h = rr - z0
    # The field returns nothing above the roof threshold, so it is unmeasured in the roof raster (which is filled
    # everywhere); a measured low cell counts as field too.
    low = inside & (~known | (np.isfinite(h) & (h < 4.)))
    lab, n = ndimage.label(low)
    if n == 0: return None
    sizes = ndimage.sum(low, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
    field = lab == k
    if field.sum() * step * step < minimum_field_m2: return None
    ring_depth = min(25., .25 * np.sqrt(footprint.area))
    # The ring is measured from the street-side outline only: the edge that faces the field is the bowl's lower
    # edge, not an outer wall. Outside cells next to the field do not seed the distance.
    near_field = ndimage.binary_dilation(field, iterations=max(1, int(round(12. / step))))
    outer_seed = ~inside & ~near_field
    if not outer_seed.any(): outer_seed = ~inside
    dist_out = ndimage.distance_transform_edt(~outer_seed) * step
    # The concourse ring takes about 40 % of the stand depth (outer outline to field), within 10-25 m.
    d_to_field = ndimage.distance_transform_edt(~field) * step
    stand = inside & ~field
    if stand.any(): ring_depth = float(np.clip(.4 * np.median((dist_out + d_to_field)[stand]), 10., 25.))
    ring = inside & (dist_out <= ring_depth) & ~field
    ring_h = float(np.nanpercentile(h[ring & known & np.isfinite(h)], 60)) if (ring & known).sum() >= 10 else 12.
    ring_h = max(ring_h, 8.)
    d_field = ndimage.distance_transform_edt(~field) * step
    bowl = inside & ~ring & ~field
    t = d_field / np.maximum(d_field + np.maximum(dist_out - ring_depth, .1), 1e-6)
    z_rel = np.floor(np.clip(t, 0, 1) * ring_h / row_rise_m) * row_rise_m
    bowl_z = np.where(bowl, z0 + z_rel, np.nan)
    ring_poly = cells_area(ring, footprint, xmin, ymax, step).buffer(0)
    return {'ring': ring_poly, 'field': cells_area(field, footprint, xmin, ymax, step).buffer(0), 'ringTopZ': z0 + ring_h, 'ringDepthM': ring_depth, 'bowlMask': bowl, 'bowlZ': bowl_z,
            'fieldM2': float(field.sum() * step * step)}


def side_materials(mesh,records,default):
    """Faces of a roof solid's vertical sides grouped by the material of the wall each stands over: within .6 m of the
    wall's plane, inside its length, facing the same way. A frontage drawn as an envelope graph gives its top node's
    material. Faces over no wall keep `default`; faces over a gable end drawn to its ridge (the wall fills them) go under
    None, not to be drawn."""
    walls=[]
    for r in records:
        mat=r.get('material') or (max(r['nodes'],key=lambda d:d['rect'][1]+d['rect'][3]).get('material') if r.get('nodes') else None)
        if not (mat or r.get('gableEnd')) or r.get('midpoint') is None or r.get('normal') is None or not r.get('lengthM'):continue
        n=np.asarray(r['normal'][:2],float);L=np.linalg.norm(n)
        if L<1e-6:continue
        n=n/L;walls.append((np.asarray(r['midpoint'][:2],float),n,np.array([-n[1],n[0]]),float(r['lengthM'])/2+.3,None if r.get('gableEnd') else mat))
    out={}
    for i,(p,q) in enumerate(zip(mesh.triangles_center,mesh.face_normals)):
        key=default
        for m,n,t,half,mat in walls:
            if float(q[:2]@n)<.9:continue
            d=p[:2]-m
            if abs(float(d@n))<=.6 and abs(float(d@t))<=half:key=mat;break
        out.setdefault(key,[]).append(i)
    return out
