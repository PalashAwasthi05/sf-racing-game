"""Explicit, measured recesses in a facade wall.

A recess is a negative wall volume with visible back and return surfaces. It is
kept separate from balconies: a balcony projects in front of an intact wall,
while a recess removes part of that wall. Dimensions are profile fractions in
the compiled profile and are clipped to the wall piece that owns them.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import triangulate, unary_union


def compile_reading(reading, fx, fz):
    """Convert explicitly observed metric reading bounds into profile fractions."""
    result = []
    for item in reading.get('recesses') or []:
        try:
            x0, z0, x1, z1 = (float(item[k]) for k in ('x0', 'z0', 'x1', 'z1'))
            depth = float(item['depthM'])
        except (KeyError, TypeError, ValueError):
            continue
        if x1 <= x0 or z1 <= z0 or not .2 <= depth <= 8.:
            continue
        rec = {'rect': [fx(x0), fz(z0), fx(x1), fz(z1)], 'depthM': depth}
        if item.get('kind') in ('loggia', 'recessed_terrace', 'recessed_entry'):
            rec['kind'] = item['kind']
        for key in ('material', 'backMaterial', 'returnMaterial', 'soffitMaterial', 'floorMaterial'):
            if isinstance(item.get(key), str):
                rec[key] = item[key]
        rail = item.get('frontRailing')
        if isinstance(rail, dict) and rail.get('kind') in ('metal', 'balustrade', 'solid'):
            try:
                rh = float(rail['heightM'])
                if not .4 <= rh <= 1.6: raise ValueError
                clean_rail = {'kind': rail['kind'], 'heightM': rh,
                              'horizontalRails': max(1, min(4, int(rail.get('horizontalRails', 2)))),
                              'material': rail.get('material', '#e4e2da')}
                pitch = rail.get('postPitchM')
                if pitch is not None and .25 <= float(pitch) <= 2.:
                    clean_rail['postPitchM'] = float(pitch)
                rec['frontRailing'] = clean_rail
            except (KeyError, TypeError, ValueError):
                pass
        back = []
        for op in item.get('backOpenings') or []:
            try:
                ox0, oz0, ox1, oz1 = (float(op[k]) for k in ('x0', 'z0', 'x1', 'z1'))
            except (KeyError, TypeError, ValueError):
                continue
            if not (x0 <= ox0 < ox1 <= x1 and z0 <= oz0 < oz1 <= z1):
                continue
            q = {'rect': [fx(ox0), fz(oz0), fx(ox1), fz(oz1)],
                 'kind': op.get('kind') if op.get('kind') in ('window', 'glazed_door', 'door') else 'window'}
            for key in ('material', 'frameMaterial'):
                if isinstance(op.get(key), str): q[key] = op[key]
            if isinstance(op.get('lites'), list) and len(op['lites']) == 2:
                q['lites'] = [max(1, min(8, int(v))) for v in op['lites']]
            back.append(q)
        if back: rec['backOpenings'] = back
        result.append(rec)
    return result


def profile_regions(profile, W, Hs, H, low, high, clip_box, bays=()):
    """Return valid wall-frame recess rectangles, omitting any that cross a bay.

    Bay bodies are independent solids with their own openings. The host-wall
    recess compiler leaves an intersecting feature intact for review rather
    than cutting through that separately owned geometry.
    """
    excluded = []
    for bay in bays or ():
        try: excluded.append(box(float(bay['x0']), float(bay['z0']), float(bay['x1']), float(bay['z1'])))
        except (KeyError, TypeError, ValueError): continue
    out = []
    for item in profile.get('recesses') or []:
        try:
            x0, z0, x1, z1 = map(float, item['rect'])
            x0, x1, z0, z1 = x0 * W, x1 * W, z0 * Hs, z1 * Hs
            depth = float(item['depthM'])
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 < .25 or z1 - z0 < .25 or not .2 <= depth <= 8.:
            continue
        original = box(x0, z0, x1, z1)
        if any(original.intersection(b).area > 1e-5 for b in excluded):
            continue
        xa, xb = max(low, x0), min(high, x1)
        _, clip_z0, _, clip_z1 = clip_box.bounds
        za, zb = max(float(clip_z0), z0), min(H, float(clip_z1), z1)
        if xb - xa < .25 or zb - za < .25: continue
        rect = box(xa, za, xb, zb)
        if not clip_box.covers(rect): continue
        out.append({**item, 'bounds': (xa, za, xb, zb), 'depthM': depth,
                    'leftReturn': abs(xa - x0) <= 1e-6, 'rightReturn': abs(xb - x1) <= 1e-6})
    return out


def cutout(regions):
    """Union of the explicitly observed opening volumes to subtract from host wall."""
    return unary_union([box(*r['bounds']) for r in regions]) if regions else Polygon()


def remove_host_wall(wall_region, regions):
    """Subtract observed recesses from a facade cell before it is extruded."""
    void = cutout(regions)
    return wall_region if void.is_empty else wall_region.difference(void)


def _mesh(ctx, points, material, key):
    import trimesh
    pts = np.asarray(points, float)
    if len(pts) < 3: return
    faces = [[0, i, i + 1] for i in range(1, len(pts) - 1)]
    mesh = trimesh.Trimesh(vertices=pts, faces=np.asarray(faces, dtype=np.int64), process=False)
    if len(mesh.faces): ctx['put'](mesh, key, material)


def _surface_polygon(ctx, poly, depth, material, key):
    """Triangulate an elevation-plane polygon at one measured depth."""
    if poly.is_empty: return
    for tri in triangulate(poly):
        if not poly.covers(tri): continue
        coords = list(tri.exterior.coords)[:3]
        _mesh(ctx, [ctx['point'](x, z, depth) for x, z in coords], material, key)


def _quad(ctx, points, material, key, reverse=False):
    p = list(points)
    if reverse: p.reverse()
    _mesh(ctx, p, material, key)


def draw(ctx, regions, key):
    """Draw cavity back, sides, soffit and floor without inventing thickness.

    `point(x,z,depth)` maps a facade point into the world. Sheet surfaces are
    intentional: no structural thickness is synthesized when the reading did
    not measure one. Back openings are cut only when explicitly supplied.
    """
    for rec in regions:
        x0, z0, x1, z1 = rec['bounds']; d = float(rec['depthM'])
        # A cavity belongs to its local facade zone. The building-wide dominant
        # finish can be brick while this particular loggia is painted render.
        # An explicit cavity observation still takes precedence over inheritance.
        inherited = (ctx['wall_mat']((x0 + x1) / 2, (z0 + z1) / 2)
                     if callable(ctx.get('wall_mat')) else ctx['mat'])
        base = rec.get('material', inherited)
        back_mat = rec.get('backMaterial', base)
        return_mat = rec.get('returnMaterial', base)
        soffit_mat = rec.get('soffitMaterial', base)
        floor_mat = rec.get('floorMaterial', base)
        back_rect = box(x0, z0, x1, z1)
        openings = []
        for op in rec.get('backOpenings') or []:
            try:
                a, b, c, e = map(float, op['rect'])
                a, c, b, e = a * ctx['W'], c * ctx['W'], b * ctx['Hs'], e * ctx['Hs']
                opening = box(a, b, c, e).intersection(back_rect)
            except (KeyError, TypeError, ValueError):
                continue
            if opening.is_empty or opening.area < .01: continue
            openings.append((op, opening))
        back = back_rect.difference(unary_union([g for _, g in openings]) if openings else Polygon())
        _surface_polygon(ctx, back, -d, back_mat, key)

        # The side panels face the loggia interior. Only source-measured outer
        # x edges get returns; clipping at another wall segment adds no jamb.
        if rec.get('leftReturn'):
            _quad(ctx, [ctx['point'](x0, z0, 0), ctx['point'](x0, z0, -d),
                        ctx['point'](x0, z1, -d), ctx['point'](x0, z1, 0)], return_mat, key)
        if rec.get('rightReturn'):
            _quad(ctx, [ctx['point'](x1, z0, 0), ctx['point'](x1, z0, -d),
                        ctx['point'](x1, z1, -d), ctx['point'](x1, z1, 0)], return_mat, key, reverse=True)
        # x cross inward-depth is +z, so reverse the ceiling and keep the floor.
        _quad(ctx, [ctx['point'](x0, z1, 0), ctx['point'](x1, z1, 0),
                    ctx['point'](x1, z1, -d), ctx['point'](x0, z1, -d)], soffit_mat, key, reverse=True)
        _quad(ctx, [ctx['point'](x0, z0, 0), ctx['point'](x1, z0, 0),
                    ctx['point'](x1, z0, -d), ctx['point'](x0, z0, -d)], floor_mat, key)

        rail = rec.get('frontRailing')
        if rail and ctx.get('block'):
            rh = float(rail['heightM']); rm = rail.get('material', return_mat)
            width = x1 - x0; cx = (x0 + x1) / 2; top = z0 + rh
            if rail.get('kind') == 'solid':
                ctx['block'](cx, z0 + rh / 2, width, rh, .08, rm, .02, key)
            else:
                nrails = int(rail.get('horizontalRails', 2))
                for i in range(nrails):
                    zz = z0 + rh * (i + 1) / nrails
                    ctx['block'](cx, zz, width, .035, .035, rm, .02, key)
                pitch = rail.get('postPitchM')
                xs = [x0 + .03, x1 - .03]
                if pitch:
                    xs += list(np.arange(x0 + pitch, x1 - .02, pitch))
                for x in sorted(set(round(float(q), 5) for q in xs)):
                    ctx['block'](x, z0 + rh / 2, .025, rh, .035, rm, .02, key)

        for op, opening in openings:
            a, b, c, e = opening.bounds
            material = op.get('material', 'glass' if op.get('kind') in ('window', 'glazed_door') else back_mat)
            _surface_polygon(ctx, opening, -d + .01, material, key)
            fw = .045
            frame = op.get('frameMaterial', return_mat)
            # Four measured-boundary frame members; no inferred pane divisions.
            for xa, xb, za, zb in ((a, c, b, b + fw), (a, c, e - fw, e),
                                   (a, a + fw, b + fw, e - fw), (c - fw, c, b + fw, e - fw)):
                if xb > xa and zb > za:
                    _surface_polygon(ctx, box(xa, za, xb, zb), -d + .02, frame, key)
            nc, nr = (op.get('lites') or [1, 1])
            if nc > 1:
                for i in range(1, nc):
                    x = a + (c - a) * i / nc
                    _quad(ctx, [ctx['point'](x - .012, b, -d + .025), ctx['point'](x + .012, b, -d + .025),
                                ctx['point'](x + .012, e, -d + .025), ctx['point'](x - .012, e, -d + .025)], frame, key)
            if nr > 1:
                for i in range(1, nr):
                    z = b + (e - b) * i / nr
                    _quad(ctx, [ctx['point'](a, z - .012, -d + .025), ctx['point'](c, z - .012, -d + .025),
                                ctx['point'](c, z + .012, -d + .025), ctx['point'](a, z + .012, -d + .025)], frame, key)
