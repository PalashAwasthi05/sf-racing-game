"""Facade bands swept over the measured support profile at their elevation.

Coordinates are in a wall-local plan frame: ``x`` runs along the facade and
``d`` points out from it.  A band asks for its support path at the band
elevation, so bays that are active there carry the band around their actual
square, angled, or curved plan profile.  Relief placement can use the same
path with :func:`support_depth_at`.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


def support_path(x0, x1, z, bays=()):
    """Return the facade's exposed plan boundary at elevation ``z``.

    The returned ``LineString`` travels from ``(x0, 0)`` to ``(x1, 0)`` and
    detours around every active bay. Bay ``z0``/``z1`` are the elevations of
    their wall faces; a band outside that interval remains on the flat wall.
    Overlapping bay footprints are unioned before the outer support is read.
    """
    x0, x1, z = float(x0), float(x1), float(z)
    if not np.isfinite([x0, x1, z]).all() or x1 <= x0:
        raise ValueError("band range and elevation must be finite with x1 > x0")

    eps = max(1e-6, (x1 - x0) * 1e-8)
    footprint = box(x0, -eps, x1, 0.)
    active_any = False
    # Local import avoids making the bay renderer depend on this module.
    from . import bays as bay_geometry

    for bay in bays or ():
        try:
            bx0, bx1 = float(bay["x0"]), float(bay["x1"])
            bz0, bz1 = float(bay["z0"]), float(bay["z1"])
            projection = float(bay["p"])
            angle = float(bay.get("sideAngleDeg") or 45.)
        except (KeyError, TypeError, ValueError):
            continue
        if (not np.isfinite([bx0, bx1, bz0, bz1, projection, angle]).all()
                or bx1 <= bx0 or projection <= 0 or not (bz0 - 1e-8 <= z <= bz1 + 1e-8)
                or bx1 <= x0 or bx0 >= x1):
            continue

        bx0, bx1 = max(x0, bx0), min(x1, bx1)
        shape, points = bay_geometry.corners(
            float(bay["x0"]), float(bay["x1"]), projection,
            bay.get("shape"), angle,
        )
        if shape == "curved":
            path_points = [(float(bay["x0"]), 0.), *points, (float(bay["x1"]), 0.)]
        else:
            path_points = points
        bay_poly = Polygon(path_points)
        if not bay_poly.is_valid:
            bay_poly = bay_poly.buffer(0)
        if bay_poly.is_empty:
            continue
        # Cropping a band to only part of a bay introduces the same vertical
        # end cut as cropping the measured wall itself.
        crop = box(bx0, -eps, bx1, projection + eps)
        clipped = bay_poly.intersection(crop)
        if not clipped.is_empty:
            footprint = unary_union([footprint, clipped])
            active_any = True

    if not active_any:
        return LineString([(x0, 0.), (x1, 0.)])

    # Select the upper exterior chain of the wall/bay union.  The thin baseline
    # rectangle contributes the (x0,0)--(x1,0) endpoints and lets disjoint bays
    # remain one connected path.
    if footprint.geom_type != "Polygon":
        polys = [g for g in getattr(footprint, "geoms", []) if g.geom_type == "Polygon"]
        if not polys:
            return LineString([(x0, 0.), (x1, 0.)])
        footprint = max(polys, key=lambda g: g.area)
    coords = np.asarray(footprint.exterior.coords, dtype=float)
    start_i = int(np.argmin(np.linalg.norm(coords - [x0, 0.], axis=1)))
    end_i = int(np.argmin(np.linalg.norm(coords - [x1, 0.], axis=1)))

    def walk(step):
        out = [coords[start_i]]
        i = start_i
        while i != end_i and len(out) <= len(coords) + 1:
            i = (i + step) % (len(coords) - 1)
            out.append(coords[i])
        return np.asarray(out)

    candidates = [walk(1), walk(-1)]
    # The exposed chain is the one with non-negative depth; the other route
    # follows the tiny closing edge below the facade line.
    def path_score(points):
        if len(points) < 2:
            return -np.inf
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        mids = (points[:-1, 1] + points[1:, 1]) / 2
        valid = mids >= -eps * 2
        return float(np.sum(lengths[valid] * np.maximum(mids[valid], 0.)))

    selected = max(candidates, key=path_score)
    # Normalize endpoints and remove redundant consecutive vertices.
    selected[0] = [x0, 0.]
    selected[-1] = [x1, 0.]
    keep = np.r_[True, np.linalg.norm(np.diff(selected, axis=0), axis=1) > 1e-8]
    selected = selected[keep]
    if len(selected) < 2:
        return LineString([(x0, 0.), (x1, 0.)])
    return LineString(selected)


def support_depth_at(path, x):
    """Return the exposed facade depth at a local x coordinate.

    At a bay's side return, the exterior support includes the side's full
    depth; taking the maximum intersection also works on a band footprint
    polygon, anchoring attached reliefs to the exact surface the sweep creates.
    """
    x = float(x)
    if not np.isfinite(x):
        return None
    bounds = path.bounds
    if not bounds or x < bounds[0] - 1e-9 or x > bounds[2] + 1e-9:
        return None
    probe = LineString([(x, bounds[1] - 1.), (x, bounds[3] + 1.)])
    hit = path.intersection(probe)
    if hit.is_empty:
        # Interpolate on the closest segment when x falls between path vertices.
        # Polygon surfaces usually intersect the probe, but a tolerance miss
        # at a boundary should still interpolate on their boundary segments.
        lines = []
        if path.geom_type in ("LineString", "LinearRing"):
            lines = [path]
        elif path.geom_type == "Polygon":
            lines = [path.exterior, *path.interiors]
        else:
            for part in getattr(path, "geoms", []):
                if part.geom_type in ("LineString", "LinearRing"):
                    lines.append(part)
                elif part.geom_type == "Polygon":
                    lines.extend([part.exterior, *part.interiors])
        for ring in lines:
            coords = np.asarray(ring.coords, dtype=float)
            for a, b in zip(coords[:-1], coords[1:]):
                if min(a[0], b[0]) - 1e-9 <= x <= max(a[0], b[0]) + 1e-9:
                    dx = b[0] - a[0]
                    if abs(dx) < 1e-9:
                        return float(max(a[1], b[1]))
                    t = np.clip((x - a[0]) / dx, 0., 1.)
                    return float(a[1] + t * (b[1] - a[1]))
        return None

    def ys(g):
        if g.geom_type == "Point":
            return [g.y]
        if g.geom_type in ("LineString", "LinearRing"):
            return [float(c[1]) for c in g.coords]
        return [y for part in getattr(g, "geoms", []) for y in ys(part)]

    values = ys(hit)
    return float(max(values)) if values else None


def should_emit_default_plinth(profile):
    """Only unsourced legacy profiles may receive the generic facade plinth."""
    return not bool((profile or {}).get("sourceIds"))


def bay_cap_replaced_by_band(bay, bands, frame_width, frame_height, tolerance=.02):
    """Whether a measured band owns a bay's decorative cornice region.

    The bay renderer still closes the cladding with its thin structural lid.
    When a source-measured course spans the full bay and overlaps the two
    decorative cornice layers, those inferred layers are redundant and should
    be omitted. Bands below or above the cap, or bands covering only part of
    the bay, remain separate geometry.
    """
    if (bay or {}).get("cap", "cornice") != "cornice":
        return False
    x0, x1 = float(bay["x0"]), float(bay["x1"])
    z1 = float(bay["z1"])
    width, height = float(frame_width), float(frame_height)
    cap_lo, cap_hi = z1 + .14, z1 + .34
    for band in bands or ():
        xr = band.get("xRange", [0., 1.])
        if len(xr) != 2:
            continue
        bx0, bx1 = sorted((float(xr[0]) * width, float(xr[1]) * width))
        if bx0 > x0 + tolerance or bx1 < x1 - tolerance:
            continue
        center = float(band.get("height", 0.)) * height
        if band.get("section"):
            extents = [(center + float(rise) - float(hh) / 2.,
                        center + float(rise) + float(hh) / 2.)
                       for rise, hh, _projection in band["section"]]
        else:
            thickness = float(band.get("thicknessM", .16))
            extents = [(center - thickness / 2., center + thickness / 2.)]
        if any(top > cap_lo + 1e-8 and bottom < cap_hi - 1e-8
               for bottom, top in extents):
            return True
    return False


def _plan_polygons(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return [g for part in getattr(geometry, "geoms", []) for g in _plan_polygons(part)]


def band_footprint(path, projection, depth):
    """Return the plan polygon occupied by a swept band.

    This is also the exact input used by :func:`sweep_band`; passing it to
    relief placement and querying it with :func:`support_depth_at` keeps relief
    supports co-located with the bay-following band face.
    """
    projection, depth = float(projection), float(depth)
    if not np.isfinite([projection, depth]).all() or depth <= .002:
        return Polygon()
    if abs(projection) > 1e-8:
        shifted = path.parallel_offset(abs(projection), "left" if projection > 0 else "right", join_style=2)
    else:
        shifted = path
    return shifted.buffer(depth / 2., cap_style=2, join_style=2)


def sweep_band(point, x0, x1, z, height, depth, projection=0., bays=(), engine="earcut"):
    """Create one closed band mesh swept over its elevation-specific support.

    ``point(x, z, d)`` maps a facade-local coordinate to world XYZ. ``depth``
    is the full horizontal band thickness; ``projection`` shifts its centerline
    outward from the support, matching the existing facade-block convention.
    Return a list because a self-intersecting/concave support may separate into
    multiple valid plan polygons.
    """
    import trimesh

    height, depth, projection = map(float, (height, depth, projection))
    if not np.isfinite([height, depth, projection]).all() or min(height, depth) <= .002:
        return []
    path = support_path(x0, x1, z + height / 2., bays)
    footprint = band_footprint(path, projection, depth)
    polys = _plan_polygons(footprint)
    origin = np.asarray(point(0., z, 0.), dtype=float)
    x_axis = np.asarray(point(1., z, 0.), dtype=float) - origin
    d_axis = np.asarray(point(0., z, 1.), dtype=float) - origin
    z_axis = np.asarray(point(0., z + 1., 0.), dtype=float) - origin
    transform = np.eye(4)
    transform[:3, 0] = x_axis
    transform[:3, 1] = d_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    meshes = []
    for poly in polys:
        if poly.is_empty or not poly.is_valid or poly.area < .001:
            continue
        mesh = trimesh.creation.extrude_polygon(poly, height=height, engine=engine)
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def sweep_profile_band(point, source_point, frame_x0, frame_x1, segment_x0,
                       segment_x1, z, height, depth, projection=0., bays=(),
                       emitted=None, identity=None, engine="earcut"):
    """Sweep this wall segment's flat band and emit each bay wrap once.

    Wall outlines can partition a measured bay into a host edge, a mapped
    front edge, and short returns. Flat band portions still belong to the
    current edge plane. The portion following an observed bay belongs to the
    reading's source plane and spans the entire bay, independent of which edge
    happened to be compiled first.
    """
    frame_x0, frame_x1, segment_x0, segment_x1 = map(
        float, (frame_x0, frame_x1, segment_x0, segment_x1))
    z, height = float(z), float(height)
    if not np.isfinite([frame_x0, frame_x1, segment_x0, segment_x1, z, height]).all():
        return []
    band_lo, band_hi = sorted((frame_x0, frame_x1))
    seg_lo, seg_hi = sorted((segment_x0, segment_x1))
    seg_lo, seg_hi = max(seg_lo, band_lo), min(seg_hi, band_hi)
    if band_hi - band_lo <= .002 or height <= .002:
        return []

    center_z = z + height / 2.
    active = []
    for index, bay in enumerate(bays or ()):
        try:
            bx0, bx1 = sorted((float(bay['x0']), float(bay['x1'])))
            bz0, bz1 = sorted((float(bay['z0']), float(bay['z1'])))
        except (KeyError, TypeError, ValueError):
            continue
        # A bay's measured decorative cornice occupies the 0.14--0.34 m band
        # immediately above its cladding lid. Let an observed course in that
        # zone follow the same plan outline instead of reverting to the
        # segmented host-edge planes.
        # Even when a measured band replaces the decorative cornice, it still
        # needs the same supported plan outline through that former cap zone.
        cap_z1 = bz1 + .34
        if bz0 - 1e-8 <= center_z <= cap_z1 + 1e-8:
            lo, hi = max(band_lo, bx0), min(band_hi, bx1)
            if hi - lo > .002:
                support_bay = bay
                if center_z > bz1:
                    support_bay = {**bay, "z1": cap_z1}
                active.append((lo, hi, support_bay, bay.get('index', index)))

    meshes = []
    # Keep ordinary wall spans on their selected physical edge plane.
    intervals = [(seg_lo, seg_hi)] if seg_hi - seg_lo > .002 else []
    # Bay dictionaries are not orderable. More importantly, equal horizontal
    # bounds do not make two vertically distinct bays interchangeable. Sort
    # by stable scalar geometry and identity fields, retaining both records.
    active.sort(key=lambda item: (item[0], item[1],
                                  float(item[2]['z0']), float(item[2]['z1']),
                                  str(item[3])))
    for lo, hi, _, _ in active:
        split = []
        for a, b in intervals:
            if hi <= a + .002 or lo >= b - .002:
                split.append((a, b))
                continue
            if lo - a > .002:
                split.append((a, lo))
            if b - hi > .002:
                split.append((hi, b))
        intervals = split
    for lo, hi in intervals:
        meshes.extend(sweep_band(point, lo, hi, z, height, depth, projection,
                                 bays=(), engine=engine))

    # A bay-following cap is one continuous measured assembly, not a series of
    # partial caps recreated on every mapped footprint facet.
    if emitted is None:
        emitted = set()
    for lo, hi, bay, bay_index in active:
        key = (identity, bay_index) if identity is not None else bay_index
        if key in emitted:
            continue
        meshes.extend(sweep_band(source_point or point, lo, hi, z, height,
                                 depth, projection, bays=[bay], engine=engine))
        emitted.add(key)
    return meshes


def sweep_section(point, x0, x1, z, section, bays=(), engine="earcut"):
    """Sweep a cornice section ``[(rise, height, projection), ...]``.

    Each section element uses the same bay support evaluated at its own
    elevation, so relief anchors can query that exact section with
    :func:`support_path` and :func:`support_depth_at`.
    """
    meshes = []
    for rise, height, projection in section or ():
        rise, height, projection = map(float, (rise, height, projection))
        # ``z`` is the section's centerline elevation. Keep the same center
        # convention used by the facade grammar's section sweep.
        meshes.extend(sweep_band(point, x0, x1, z + rise - height / 2., height,
                                 projection + .02, projection / 2., bays, engine))
    return meshes
