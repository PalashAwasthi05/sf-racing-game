"""Dormers and chimneys on a roof, from a given massing: what stands on a roof that the roof form does not hold.

A massing entry's `roofFeatures` (areas/ring/massing-overrides.json, read from the photographs and checked against
the LiDAR's bumps above the roof) lists them, in the world's local frame, heights above the building's foot:
  {"kind": "dormer", "xy": [x, y] (the dormer's centre), "facing": [dx, dy] (the way its window looks), "widthM",
   "depthM", "eaveM", "ridgeM" (its own eave and ridge), "roof": "gable" | "hip" | "shed", "window": true}
An optional `frontEvidence` record can describe the visible front silhouette and typed aperture polygons in
normalized front coordinates. Rectangular legacy windows remain supported. It constrains only that visible
elevation; the inward depth remains a separate prior unless independently measured.
  {"kind": "chimney", "xy": [x, y], "widthM", "depthM", "topM", "angleDeg" (optional: its long side's bearing from east;
   default the part's long axis), "material": "brick" | "wall"}
A dormer is a small house on the slope: walls from inside the main roof up to its eave, its own roof form
(pipeline/roof_forms.py) with its ridge running back into the main roof, a window in its front. A chimney is a stack
from inside the roof to its top, with a projecting cap. Location-free: the entries carry the facts.
`build` returns (mesh, role) pairs; role is 'roof', 'wall', 'trim', 'brick' or 'glass'."""
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely import affinity
from general_world.roof_planes import roof_solid, evaluate as roof_evaluate
from pipeline import roof_forms


def _rect(c, u, w, d):
    """Rectangle centred at c, width w along unit u, depth d along u's left normal."""
    n = np.array([-u[1], u[0]]); c = np.asarray(c, float)
    return Polygon([c - u * w / 2 - n * d / 2, c + u * w / 2 - n * d / 2, c + u * w / 2 + n * d / 2, c - u * w / 2 + n * d / 2])


def roof_z(xy, patches, base_roof):
    """The roof's height at points (n, 2): the given roof form where a patch covers them, else the flat roof."""
    xy = np.atleast_2d(np.asarray(xy, float)); z = np.full(len(xy), np.nan)
    for p in patches:
        g = p['geometry'].buffer(.05); inside = np.array([g.contains(Point(q)) for q in xy])
        if inside.any(): z[inside] = np.fmax(z[inside], roof_evaluate(p['model'], xy[inside]))
    return np.where(np.isfinite(z), z, float(base_roof))


def _box(c, u, w, h, t, z0):
    """A thin slab on a wall plane: centre c (x, y), along unit u, width w, height h, thickness t (outward), bottom z0."""
    n = np.array([u[1], -u[0]])                    # u's right normal
    m = trimesh.creation.box(extents=[w, t, h])
    R = np.eye(4); R[:2, 0] = u; R[:2, 1] = n; R[2, 2] = 1; R[:3, 3] = [c[0], c[1], z0 + h / 2]
    m.apply_transform(R); return m


def _front_evidence_meshes(evidence, center_xy, facing, width, bottom, top, wall_role, trim_role,
                           generated_front=None):
    """Build a front panel with independently described, typed apertures.

    ``apertures`` accepts ``kind=window|louver|closed`` and either an
    ``outline01`` polygon or ``rect01``. Legacy ``windows`` rectangles remain
    supported. Coordinates are across the feature width and bottom-to-top span.
    The optional front ``outline01`` replaces the front silhouette; without it,
    ``generated_front`` is the actual generated shell front in metric coordinates.
    Invalid explicit evidence raises ValueError; the caller records an unresolved
    front and suppresses generic windows rather than replacing source evidence.
    """
    if not isinstance(evidence, dict) or evidence.get('coordinateSpace') != 'normalized-front':
        raise ValueError('front evidence requires normalized-front coordinateSpace')
    if not np.isfinite([width, bottom, top]).all() or width <= 0 or top <= bottom:
        raise ValueError('invalid front coordinate frame')

    def polygon01(value, label):
        try:
            outline = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(label + ' must be a finite polygon') from error
        if (outline.ndim != 2 or outline.shape[1] != 2 or len(outline) < 3
                or not np.isfinite(outline).all() or np.any(outline < 0.) or np.any(outline > 1.)):
            raise ValueError(label + ' must be a polygon within [0,1]')
        poly = Polygon(outline)
        if not poly.is_valid or poly.area < 1e-8:
            raise ValueError(label + ' must be a valid nonempty polygon')
        return poly

    def metric(poly):
        from shapely.ops import transform
        return transform(lambda x, y, z=None: ((np.asarray(x) - .5) * width,
                                               bottom + np.asarray(y) * (top - bottom)), poly)

    if 'outline01' in evidence:
        silhouette = metric(polygon01(evidence['outline01'], 'front outline01'))
    else:
        silhouette = generated_front
        if silhouette is None or silhouette.is_empty or not silhouette.is_valid:
            raise ValueError('aperture-only evidence requires a generated front silhouette')
    if 'apertures' in evidence and 'windows' in evidence:
        raise ValueError('provide apertures or legacy windows, not both')
    legacy = 'apertures' not in evidence
    items = evidence.get('windows' if legacy else 'apertures')
    if not isinstance(items, list):
        raise ValueError('front evidence requires an explicit aperture/window list')
    records = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('aperture must be an object')
        kind = item.get('kind', 'window' if legacy else None)
        if kind not in ('window', 'louver', 'closed'):
            raise ValueError('unsupported aperture kind')
        if 'outline01' in item and 'rect01' in item:
            raise ValueError('provide aperture outline01 or rect01, not both')
        if 'outline01' in item:
            opening = metric(polygon01(item['outline01'], 'aperture outline01'))
        else:
            try:
                rect = np.asarray(item.get('rect01'), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('aperture rect01 must be finite') from error
            if (rect.shape != (4,) or not np.isfinite(rect).all()
                    or not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1)):
                raise ValueError('aperture rect01 must be within [0,1]')
            opening = metric(box(*rect))
        if not silhouette.buffer(1e-7).covers(opening):
            raise ValueError('aperture lies outside the front silhouette')
        if any(opening.intersection(previous['opening']).area > 1e-8 for previous in records):
            raise ValueError('apertures overlap')
        vertical = item.get('verticalMullions01', [])
        horizontal = item.get('horizontalMullions01', [])
        if (not isinstance(vertical, list) or not isinstance(horizontal, list)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) or not 0 < v < 1 for v in vertical + horizontal)):
            raise ValueError('mullion fractions must lie within (0,1)')
        if kind != 'window' and (vertical or horizontal):
            raise ValueError('mullions require a window aperture')
        x0, y0, x1, y1 = opening.bounds
        try:
            frame = float(item.get('frameWidthM', min(.055, (x1 - x0) * .15, (y1 - y0) * .15)))
        except (TypeError, ValueError) as error:
            raise ValueError('frameWidthM must be positive') from error
        if not np.isfinite(frame) or frame <= 0:
            raise ValueError('frameWidthM must be positive')
        inner = opening.buffer(-frame, join_style=2)
        if inner.is_empty or not inner.is_valid:
            raise ValueError('frameWidthM leaves no aperture interior')
        blade_count, blade_pitch = item.get('bladeCount'), item.get('bladePitchM')
        if blade_count is not None and (isinstance(blade_count, bool)
                or not isinstance(blade_count, (int, np.integer)) or not 1 <= blade_count <= 1024):
            raise ValueError('bladeCount must be an integer from 1 to 1024')
        if blade_pitch is not None:
            try:
                blade_pitch = float(blade_pitch)
            except (TypeError, ValueError) as error:
                raise ValueError('bladePitchM must be positive') from error
            if not np.isfinite(blade_pitch) or blade_pitch <= 0 or (y1 - y0) / blade_pitch > 1024:
                raise ValueError('bladePitchM must be positive and resolvable')
        if blade_count is not None and blade_pitch is not None:
            raise ValueError('provide bladeCount or bladePitchM, not both')
        if kind != 'louver' and (blade_count is not None or blade_pitch is not None):
            raise ValueError('blade evidence requires a louver aperture')
        records.append(dict(opening=opening, inner=inner, kind=kind, vertical=vertical,
                            horizontal=horizontal, blade_count=blade_count, blade_pitch=blade_pitch))

    meshes = []
    u = np.array([facing[1], -facing[0]], dtype=float)
    def emit(poly, offset, thickness, role, component):
        if poly.is_empty:
            return
        pieces = [poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))
        transform = np.eye(4, dtype=float)
        transform[:3, 0] = [u[0], u[1], 0.]
        transform[:3, 1] = [0., 0., 1.]
        transform[:3, 2] = [facing[0], facing[1], 0.]
        transform[:3, 3] = [center_xy[0] + facing[0] * offset,
                            center_xy[1] + facing[1] * offset, 0.]
        for piece in pieces:
            if piece.geom_type != 'Polygon' or piece.area < 1e-8:
                continue
            mesh = trimesh.creation.extrude_polygon(piece, height=thickness, engine='earcut')
            mesh.apply_transform(transform)
            mesh.metadata['frontEvidenceComponent'] = component
            meshes.append((mesh, role))

    holes = unary_union([r['opening'] for r in records])
    emit(silhouette.difference(holes), 0., .035, wall_role, 'front-wall')
    for record in records:
        opening, inner = record['opening'], record['inner']
        emit(opening.difference(inner), .04, .055, trim_role, 'aperture-frame')
        # Infill is recessed behind the front ring. The generated opaque front is
        # removed by the caller, so neither curved glazing nor vents retain backing.
        emit(inner, .005, .018, 'glass' if record['kind'] == 'window' else wall_role,
             'glazing' if record['kind'] == 'window' else 'nonglass-insert')
        x0, y0, x1, y1 = opening.bounds
        for fraction in record['vertical']:
            x = x0 + (x1 - x0) * float(fraction)
            emit(inner.intersection(box(x - .022, y0, x + .022, y1)), .04, .055, trim_role, 'mullion')
        for fraction in record['horizontal']:
            y = y0 + (y1 - y0) * float(fraction)
            emit(inner.intersection(box(x0, y - .022, x1, y + .022)), .04, .055, trim_role, 'mullion')
        if record['kind'] == 'louver':
            ix0, iy0, ix1, iy1 = inner.bounds
            pitch, count = record['blade_pitch'], record['blade_count']
            if count is not None:
                pitch = (iy1 - iy0) / count
                positions = iy0 + (np.arange(count) + .5) * pitch
            elif pitch is not None:
                positions = np.arange(iy0 + pitch / 2., iy1, pitch)
            else:
                positions = []  # no source pitch/count means no invented blades
            for y in positions:
                half = min(.025, pitch * .22)
                emit(inner.intersection(box(ix0, y - half, ix1, y + half)),
                     .03, .045, trim_role, 'louver-blade')
    return meshes


def _front_wall_mask(mesh, center, facing, reach):
    normals = np.asarray(mesh.face_normals)
    centers = np.asarray(mesh.triangles_center)
    return (((normals[:, :2] @ facing) > .9) & (np.abs(normals[:, 2]) <= .05)
            & (np.abs(centers[:, :2] @ facing - (center @ facing + reach)) < .15))


def _project_front_wall(mesh, center, facing, reach):
    """Project the actual generated front triangles into its metric elevation."""
    if mesh is None or not len(mesh.faces):
        return None
    triangles = np.asarray(mesh.triangles)[_front_wall_mask(mesh, center, facing, reach)]
    u = np.array([facing[1], -facing[0]])
    pieces = []
    for triangle in triangles:
        polygon = Polygon(np.c_[(triangle[:, :2] - (center + facing * reach)) @ u, triangle[:, 2]])
        if polygon.is_valid and polygon.area > 1e-8:
            pieces.append(polygon)
    return unary_union(pieces) if pieces else None


def _without_front_wall(mesh, center, facing, reach):
    """Remove the same generated front triangles used to recover its silhouette."""
    if mesh is None or not len(mesh.faces):
        return mesh
    keep = np.where(~_front_wall_mask(mesh, center, facing, reach))[0]
    return mesh.submesh([keep], append=True, repair=False) if len(keep) else None


def _front_section(f, width, bottom, top):
    evidence = f.get('frontEvidence')
    if not isinstance(evidence, dict) or evidence.get('coordinateSpace') != 'normalized-front' or 'outline01' not in evidence:
        return None
    points = np.asarray(evidence['outline01'], float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1)):
        raise ValueError('invalid source front section')
    section = Polygon(np.c_[(points[:, 0] - .5) * width, bottom + points[:, 1] * (top - bottom)])
    if not section.is_valid or section.area <= 1e-8:
        raise ValueError('invalid source front section')
    return section


def _extrude_front_section(section, center, face, depth):
    mesh = trimesh.creation.extrude_polygon(section, height=depth, engine='earcut')
    transform = np.eye(4)
    transform[:3, 0] = [face[1], -face[0], 0.]
    transform[:3, 1] = [0., 0., 1.]
    transform[:3, 2] = [-face[0], -face[1], 0.]
    transform[:2, 3] = center + face * depth / 2.
    mesh.apply_transform(transform)
    return mesh


def _dormer_geometry(f, center, face, width, depth, bottom, eave, ridge):
    try:
        section = _front_section(f, width, bottom, ridge)
    except (ValueError, TypeError):
        section = None  # explicit aperture validation still records the invalid front
    if section is not None:
        return _extrude_front_section(section, center, face, depth), depth / 2.
    rectangle = _rect(center, np.array([face[1], -face[0]]), width, depth)
    kind = f.get('roof', 'gable') if f.get('roof', 'gable') in ('gable', 'hip', 'shed') else 'gable'
    spec = {'kind': kind, 'riseM': ridge - eave, 'overhangM': .15,
            'along': 'long' if depth >= width else 'short', 'lowSide': face.tolist()}
    if kind == 'gable':
        spec['gableEdges'] = roof_forms.gable_edges(rectangle, [(face, float(center @ face + depth / 2))], near=.05)
    roof, _ = roof_forms.build(rectangle, spec, eave, eave)
    if not roof:
        return None, depth / 2.
    patch = roof[0]
    reach = float(max((np.asarray(q) - center) @ face for q in patch['geometry'].exterior.coords))
    return roof_solid(patch['geometry'], patch['model'], bottom), reach


def dormer_cut_solids(feature, z0):
    """Convex cells of the same outer body used by source-front rendering."""
    center, face = np.asarray(feature['xy'], float), np.asarray(feature['facing'], float)
    face /= np.linalg.norm(face)
    width = float(np.clip(feature.get('widthM', 2.), .8, 8.))
    depth = float(np.clip(feature.get('depthM', 2.5), .8, 10.))
    bottom, eave, ridge = [z0 + float(feature[k]) for k in ('bottomM', 'eaveM', 'ridgeM')]
    section = _front_section(feature, width, bottom, ridge)
    if section is not None:
        vertices, triangles = trimesh.creation.triangulate_polygon(section, engine='earcut')
        return [_extrude_front_section(Polygon(vertices[t]), center, face, depth) for t in triangles]
    volume, _ = _dormer_geometry(feature, center, face, width, depth, bottom, eave, ridge)
    return [volume] if volume is not None else []


def dormer(f, z0, patches, base_roof, part=None):
    out = []; c = np.asarray(f['xy'], float); face = np.asarray(f['facing'], float); face /= max(np.linalg.norm(face), 1e-6)
    u = np.array([face[1], -face[0]])               # along the dormer's front, so that its depth runs back from `face`
    w = float(np.clip(f.get('widthM', 2.), .8, 8.)); d = float(np.clip(f.get('depthM', 2.5), .8, 10.))
    eave = z0 + float(f['eaveM']); ridge = z0 + float(f.get('ridgeM', f['eaveM'] + w * .35))
    anchored = f.get('bottomM') is not None
    if not anchored:
        # Legacy, unanchored observations can move down the slope until the front clears the main roof.
        for s in np.arange(0., 4.01, .1):
            front = c + face * (d / 2 + s)
            if part is not None and part.exterior.distance(Point(front)) < .4 and s > 0: s -= .1; break
            if eave - float(roof_z(front[None], patches, base_roof)[0]) >= 1.4: break
        s = max(float(s), 0.); c = c + face * s / 2; d = d + s
    r = _rect(c, u, w, d); corners = np.asarray(r.exterior.coords)[:4]
    roof_samples = np.vstack([corners, c])
    underlying = roof_z(roof_samples, patches, base_roof)
    # Keep measured features where observed. If the host roof is already above the dormer's complete vertical extent,
    # emit nothing; shifting the feature several metres would detach it from the photo measurement.
    if anchored and float(underlying.min()) >= ridge - .05:
        return []
    bottom = z0 + float(f['bottomM']) if anchored else float(underlying.min()) - .15
    if bottom >= eave - .05:
        return []
    solid, reach = _dormer_geometry(f, c, face, w, d, bottom, eave, ridge)
    front_meshes = None
    explicit_front = 'frontEvidence' in f
    front_error = None
    def source_front(shell):
        nonlocal front_error
        if not explicit_front:
            return None
        try:
            return _front_evidence_meshes(f.get('frontEvidence'), c + face * reach, face, w,
                bottom, ridge, _role(f, 'wall', 'wallMaterial'), _role(f, 'trim', 'trimMaterial'),
                generated_front=_project_front_wall(shell, c, face, reach))
        except ValueError as error:
            front_error = str(error)
            return None

    if solid is not None:
        s = solid
        up = s.face_normals[:, 2] > .05; side = np.abs(s.face_normals[:, 2]) <= .05
        roof_mesh = s.submesh([np.where(up)[0]], append=True, repair=False)
        wall_mesh = s.submesh([np.where(side)[0]], append=True, repair=False)
        front_meshes = source_front(wall_mesh)
        if front_meshes is not None:
            wall_mesh = _without_front_wall(wall_mesh, c, face, reach)
        out += [(roof_mesh, _role(f, 'roof', 'roofMaterial'))]
        if wall_mesh is not None and len(wall_mesh.faces):
            out.append((wall_mesh, _role(f, 'wall', 'wallMaterial')))
    else:
        m = trimesh.creation.extrude_polygon(r, height=eave - bottom); m.apply_translation([0, 0, bottom]); out.append((m, _role(f, 'wall', 'wallMaterial')))
        front_meshes = source_front(m)
        if front_meshes is not None:
            out.pop()
            m = _without_front_wall(m, c, face, reach)
            if m is not None and len(m.faces):
                out.append((m, _role(f, 'wall', 'wallMaterial')))
    if explicit_front:
        if front_meshes is not None:
            out.extend(front_meshes)
        for mesh, _ in out:
            mesh.metadata['frontEvidenceStatus'] = 'unresolved' if front_error else 'source-described'
            if front_error:
                mesh.metadata['frontEvidenceError'] = front_error
        return out  # explicit source evidence always owns aperture decisions, including invalid/empty input
    count = int(np.clip(f.get('windowCount', 1 if f.get('window', True) else 0), 0, 4))
    if count:
        front = c + face * reach                     # the front wall
        sill = z0 + float(f['bottomM']) + .08 if anchored else float(roof_z(front[None], patches, base_roof)[0]) + .2
        top = eave - .15
        if top - sill > .4:
            gap = .11; b = .055; total = min(w * .78, 4.5); ww = (total - gap * (count - 1)) / count
            hh = top - sill
            for i in range(count):
                dx = (i - (count - 1) / 2) * (ww + gap)
                out.append((_box(front + u * dx + face * .01, u, ww, hh, .02, sill), 'glass'))
                for fx, bw, fz, bh in ((dx, ww + 2 * b, sill - b, b), (dx, ww + 2 * b, top, b),
                                       (dx - (ww + b) / 2, b, sill, hh), (dx + (ww + b) / 2, b, sill, hh)):
                    out.append((_box(front + u * fx + face * .03, u, bw, bh, .06, fz), _role(f, 'trim', 'trimMaterial')))
    return out


def _role(f, default, field):
    """Return the legacy role unless a feature supplies a caller-resolved material role."""
    return f.get(field) or default


def conical_cap(f, z0):
    """A closed elliptical cone and its narrow eave band, with the footprint centered at `xy`."""
    c = np.asarray(f['xy'], float)
    w = float(f.get('widthM', 2.)); d = float(f.get('depthM', w))
    face = np.asarray(f.get('facing', [0., 1.]), float)
    face /= max(float(np.linalg.norm(face)), 1e-9)
    u = np.array([face[1], -face[0]])
    eave = z0 + float(f['eaveM']); ridge = z0 + float(f['ridgeM'])
    if not np.isfinite([*c, w, d, eave, ridge]).all() or w <= 0 or d <= 0 or ridge <= eave + 1e-4:
        return []
    sections = 32; theta = np.arange(sections) * (2 * np.pi / sections)
    ring_xy = c + np.outer((w / 2) * np.cos(theta), u) + np.outer((d / 2) * np.sin(theta), face)
    ring = np.c_[ring_xy, np.full(sections, eave)]
    verts = np.vstack([ring, [c[0], c[1], eave], [c[0], c[1], ridge]])
    center, apex = sections, sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, apex])               # outside-facing conical roof
        faces.append([center, j, i])              # downward-facing closed underside
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    band = trimesh.creation.annulus(r_min=.96, r_max=1., height=.06, sections=sections)
    transform = np.array([[u[0] * w / 2, face[0] * d / 2, 0., c[0]],
                          [u[1] * w / 2, face[1] * d / 2, 0., c[1]],
                          [0., 0., 1., eave - .03], [0., 0., 0., 1.]])
    band.apply_transform(transform)
    return [(mesh, _role(f, 'roof', 'roofMaterial')), (band, _role(f, 'trim', 'trimMaterial'))]


def chimney(f, z0, patches, base_roof, axis):
    c = np.asarray(f['xy'], float); a = np.radians(f['angleDeg']) if f.get('angleDeg') is not None else axis
    # a stack: a LiDAR bump round one takes in the wall it rises from, so its size is held to a chimney's (1.6 m at most)
    u = np.array([np.cos(a), np.sin(a)]); w = float(np.clip(f.get('widthM', .8), .3, 1.6)); d = float(np.clip(f.get('depthM', .6), .3, 1.6))
    r = _rect(c, u, w, d); corners = np.asarray(r.exterior.coords)[:4]
    bottom = float(roof_z(corners, patches, base_roof).min()) - .2; top = z0 + float(f['topM'])
    if top - bottom < .3: return []
    role = 'brick' if f.get('material', 'brick') == 'brick' else 'wall'
    m = trimesh.creation.extrude_polygon(r, height=top - bottom); m.apply_translation([0, 0, bottom])
    cap = trimesh.creation.extrude_polygon(r.buffer(.06, join_style=2), height=.1); cap.apply_translation([0, 0, top])
    return [(m, role), (cap, role)]


def build(features, part, z0, base_roof, patches):
    """Meshes for the roof features standing on this part (their centre inside its outline)."""
    out = []; axis = None
    for f in features:
        if f.get('kind') == 'conical-cap':
            w = float(f.get('widthM', 2.)); d = float(f.get('depthM', w))
            face = np.asarray(f.get('facing', [0., 1.]), float); face /= max(float(np.linalg.norm(face)), 1e-9)
            u = np.array([face[1], -face[0]])
            ellipse = affinity.affine_transform(Point(0., 0.).buffer(1., quad_segs=8),
                [u[0] * w / 2, face[0] * d / 2, u[1] * w / 2, face[1] * d / 2,
                 float(f['xy'][0]), float(f['xy'][1])])
            if part.intersection(ellipse).area <= 1e-6: continue
            out += conical_cap(f, z0)
            continue
        if not part.contains(Point(f['xy'])): continue
        if axis is None:
            mrr = np.asarray(part.minimum_rotated_rectangle.exterior.coords)
            e = [mrr[1] - mrr[0], mrr[2] - mrr[1]]; axis = float(np.arctan2(*max(e, key=np.linalg.norm)[::-1]))
        if f.get('kind') == 'dormer' and f.get('eaveM') is not None and f.get('facing'): out += dormer(f, z0, patches, base_roof, part)
        elif f.get('kind') == 'chimney' and f.get('topM') is not None: out += chimney(f, z0, patches, base_roof, axis)
    return out
