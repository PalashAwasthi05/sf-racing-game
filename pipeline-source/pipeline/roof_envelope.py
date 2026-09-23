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


def continuous_patches(footprint,points,step,min_slope=.015,min_area_m2=25.,min_rise_m=.6):
    """Tilted roof planes the fit supports. The thresholds come from the area config's LiDAR role: a drainage
    fall of a percent or two is a flat roof, and splitting it into separately tilted planes painted the roof
    in irregular patches."""
    patches,report=fit_patches(footprint,points,RoofPolicy(cell_m=step))
    accepted=[]
    for p in patches:
        if not any(np.linalg.norm(c[:2])>=min_slope for c in p['model']['planes']):continue
        for g in ([p['geometry']] if p['geometry'].geom_type=='Polygon' else list(p['geometry'].geoms)):
            if g.area<min_area_m2 or g.buffer(-1.5).is_empty:continue
            xy=np.asarray(g.exterior.coords);z=evaluate(p['model'],xy)
            if z.max()-z.min()<min_rise_m:continue
            accepted.append({**p,'geometry':g})
    return accepted,report


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
    mask = ((grade >= minimum_grade) & (grade <= 1.0) | sloped_cells(rr, step)) & known   # smoothed grade or a raw continuous slope
    lab, n = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * step * step < minimum_area_m2: continue
        if np.nanpercentile(smooth[m], 95) - np.nanpercentile(smooth[m], 5) < minimum_rise_m: continue
        if np.median(rough[m]) > maximum_roughness_m or np.median(grade[m]) < minimum_grade: continue
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
