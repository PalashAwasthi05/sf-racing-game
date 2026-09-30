"""Wheel assemblies: tyre, rim, brake disc and caliper, modelled for the right side with the axle along +z (outward).

Tyres and the rim barrel are lathed from profiles (the tread's circumferential grooves are part of the profile), the
spoked face is an implicit solid meshed finely (five twin spokes over a concave face, a plain centre cap, five lug
nuts), the disc is a lathed ring with a hat, the caliper a rounded block over the disc edge. No markings anywhere.
"""
import numpy as np
from mesher import mesh_solid, mesh_solid_sharp, autosmooth, Mesh, merge

RIM_R = .254          # 20 inch bead seat


def lathe(profile, segments=160, sharp=()):
    """Revolve a profile of (r, z) points about the z axis. `sharp` lists profile indices where the normal breaks."""
    p = np.asarray(profile, float)
    # split at sharp corners by duplicating those points
    runs, cur = [], [0]
    for i in range(1, len(p)):
        cur.append(i)
        if i in sharp and i < len(p) - 1:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    meshes = []
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    for run in runs:
        q = p[run]
        d = np.gradient(q, axis=0)
        n2 = np.stack([d[:, 1], -d[:, 0]], 1)          # profile normal (dr, dz) rotated
        n2 /= np.maximum(np.linalg.norm(n2, axis=1), 1e-12)[:, None]
        V = np.stack([q[:, 0][:, None] * c, q[:, 0][:, None] * s, np.repeat(q[:, 1][:, None], segments, 1)], -1).reshape(-1, 3)
        N = np.stack([n2[:, 0][:, None] * c, n2[:, 0][:, None] * s, np.repeat(n2[:, 1][:, None], segments, 1)], -1).reshape(-1, 3)
        T = []
        m = len(q)
        for i in range(m - 1):
            a = i * segments + np.arange(segments)
            b = i * segments + (np.arange(segments) + 1) % segments
            a2, b2 = a + segments, b + segments
            T.append(np.stack([a, b, b2], 1))
            T.append(np.stack([a, b2, a2], 1))
        meshes.append(Mesh(V, np.concatenate(T), N))
    return merge(meshes)


def tyre_profile(R, width, grooves):
    """Closed section of a low-profile tyre from the inner bead round the sidewall, shoulder and tread to the outer bead,
    as (r, z) with z outward; the profile runs so that the lathe's normals face out of the rubber."""
    hw = width / 2
    side_top = R - .018
    pts = []
    # outer sidewall (z > 0) from the bead up to the shoulder: bulges to full width low on the wall
    for t in np.linspace(0, 1, 18):
        r = RIM_R + .004 + (side_top - RIM_R - .004) * t
        z = hw * (.90 + .10 * np.sin(np.pi * np.clip(t * 1.25, 0, 1)) ** .7) - .012 * t ** 3
        pts.append((r, z))
    # shoulder round to the tread
    zs = pts[-1][1]
    for a in np.linspace(0, np.pi / 2, 9)[1:]:
        pts.append((side_top + .018 * np.sin(a), zs - (zs - (hw - .03)) * (1 - np.cos(a)) ** .9))
    # tread with circumferential grooves, outer edge to inner edge
    z = hw - .03
    for gc, gw, gd in sorted(grooves, key=lambda g: -g[0]):
        pts += [(R, gc + gw / 2 + .002), (R - gd, gc + gw / 2), (R - gd, gc - gw / 2), (R, gc - gw / 2 - .002)]
    pts.append((R, -(hw - .03)))
    tread_end = len(pts)
    mirror = [(r, -z) for r, z in pts[:tread_end - 1 - 4 * len(grooves)][::-1]]
    pts += mirror[1:]
    return pts


def tyre(R, width):
    g = [(.055 * width / .235, .009, .006), (.018, .008, .006), (-.018, .008, .006), (-.055 * width / .235, .009, .006)]
    prof = tyre_profile(R, width, g)
    # inner surface (sits on the rim): close the section along the bead seat
    prof = prof + [(RIM_R + .004, -width / 2 * .9), (RIM_R + .004, width / 2 * .9)]
    return lathe(prof[::-1], segments=96)


def rim_barrel(width):
    """Rim flange and barrel: visible lip outboard, dark barrel inside the spokes."""
    hw = width / 2
    prof = [(RIM_R - .006, hw - .02), (RIM_R + .01, hw - .004), (RIM_R + .012, hw + .004), (RIM_R + .004, hw + .010),
            (RIM_R - .012, hw + .006), (RIM_R - .02, hw - .02), (RIM_R - .02, -hw + .03), (RIM_R + .002, -hw + .005), (RIM_R + .004, -hw)]
    return lathe(prof, segments=180, sharp=(3, 5, 6))


def disc(r_out=.175, thick=.034):
    prof = [(.10, -thick / 2), (r_out, -thick / 2), (r_out, thick / 2), (.10, thick / 2), (.10, -thick / 2)]
    ring = lathe(prof, segments=140, sharp=(1, 2, 3))
    hat = lathe([(.095, -thick / 2 + .004), (.095, .045), (.075, .052), (.04, .052)], segments=100, sharp=(1,))
    return ring, hat


def drill_holes(r_out=.175, thick=.034, n_arcs=18):
    """Cross-drilled holes on both faces: three holes along each of 18 curved arcs, drawn as dark discs."""
    parts = []
    k = 8
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    for face in (1, -1):
        for i in range(n_arcs):
            for j, r in enumerate((.118, .137, .156)):
                th = 2 * np.pi * i / n_arcs + .09 * j
                c = np.array([r * np.cos(th), r * np.sin(th), face * (thick / 2 + .0004)])
                ring = c + .0036 * np.stack([np.cos(a), np.sin(a), np.zeros(k)], 1)
                V = np.concatenate([ring, c[None]])
                T = np.stack([np.arange(k), (np.arange(k) + 1) % k, np.full(k, k)], 1)
                if face < 0:
                    T = T[:, ::-1]
                parts.append(Mesh(V, T, np.tile([0, 0, face], (k + 1, 1)).astype(float)))
    return merge(parts)


def face_solid(hw_rim, lip_z, hub_z):
    """Spoked face: five twin spokes from a ring at the lip to a hub recessed to hub_z; plain cap, five lug nuts."""
    pairs = 5

    def f(x, y, z):
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        # face height by radius: concave from the lip down to the hub
        t = np.clip((r - .088) / (.236 - .088), 0, 1)
        zf = hub_z + (lip_z - hub_z) * t ** 1.35
        # twin spokes: pair centre every 72 deg, the two blades converge toward the hub
        k = np.round(th / (2 * np.pi / pairs))
        dth = th - k * 2 * np.pi / pairs
        split = np.deg2rad(11.0 + 3.0 * t)       # ten slim spokes in five loose pairs
        d_best = None
        for sgn in (-1, 1):
            lat = r * np.sin(dth - sgn * split)
            half_w = .0068 + .0045 * (1 - t)
            depth = .015 + .008 * (1 - t)
            dz = z - (zf - depth)
            q = np.stack([np.abs(lat) - half_w, np.abs(dz) - depth], 0)
            d = np.hypot(np.maximum(q[0], 0), np.maximum(q[1], 0)) + np.minimum(np.maximum(q[0], q[1]), 0) - .003
            d_best = d if d_best is None else np.minimum(d_best, d)
        spokes = np.maximum(d_best, np.maximum(.08 - r, r - .245))
        # outer ring joining the spoke ends to the barrel
        ring_d = np.maximum(np.abs(r - .242) - .012, np.abs(z - (lip_z - .014)) - .014)
        # hub: a dished disc with a raised cap
        hub = np.maximum(r - .095, np.abs(z - (hub_z - .012)) - .014)
        cap = np.maximum(r - .036, np.abs(z - (hub_z + .004)) - .008)
        # lug nuts on a 130 mm circle
        kk = np.round(th / (2 * np.pi / 5))
        a = kk * 2 * np.pi / 5
        nx, ny = .065 * np.cos(a), .065 * np.sin(a)
        nut = np.maximum(np.hypot(x - nx, y - ny) - .0105, np.abs(z - (hub_z + .004)) - .009)
        seat = np.maximum(np.hypot(x - nx, y - ny) - .016, np.abs(z - (hub_z + .0)) - .006)
        s = np.minimum(np.minimum(spokes, ring_d), hub)
        s = np.maximum(s, -seat)          # nut seats sunk into the hub
        s = np.minimum(s, np.minimum(cap, nut))
        return s
    return f


def caliper_solid(r_disc, thick, angle, pistons_w):
    """A monobloc caliper straddling the disc edge, centred at `angle` (radians, 0 = forward)."""
    ca, sa = np.cos(angle), np.sin(angle)

    def f(x, y, z):
        # local polar about the axle, rotated so the caliper centre is at u axis
        u = x * ca + y * sa
        v = -x * sa + y * ca
        r = np.hypot(u, v)
        ang = np.arctan2(v, u)
        span = np.deg2rad(34)
        d_ang = (np.abs(ang) - span) * r
        d_r = np.abs(r - (r_disc - .012)) - .032
        d_z = np.abs(z) - (thick / 2 + pistons_w)
        q = np.stack([d_ang, d_r, d_z])
        body = np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(0), 0) - .008
        # slot the disc runs through (open toward the axle)
        slot = np.maximum(np.maximum(np.abs(z) - (thick / 2 + .003), (r_disc - .045) - r), r - (r_disc + .004))
        return np.maximum(body, -slot)
    return f


def assembly(axle):
    """Parts of one wheel as (part, finish, mesh, rotates)."""
    if axle == 'front':
        R, width, rim_w, disc_t, calip_ang = .336, .235, .216, .034, np.deg2rad(160)
    else:
        R, width, rim_w, disc_t, calip_ang = .347, .265, .267, .028, np.deg2rad(20)
    hw = rim_w / 2
    lip_z = hw + .006
    hub_z = lip_z - (.034 if axle == 'front' else .052)
    parts = [('tyre', tyre(R, width), True), ('rim', rim_barrel(rim_w), True)]
    f = face_solid(hw, lip_z, hub_z)
    # dual contouring keeps the spokes' edges straight (surface nets left them sawtoothed at this grid)
    V, T = mesh_solid_sharp(f, (-.265, -.265, hub_z - .05), (.265, .265, lip_z + .01), .003, tolerance=.0002)
    parts.append(('rim', autosmooth(Mesh(V, T), f), True))
    ring, hat = disc(thick=disc_t)
    holes = drill_holes(thick=disc_t)
    zd = hub_z - .075
    for m in (ring, hat, holes):
        m.V[:, 2] += zd
    parts += [('disc', ring, True), ('hat', hat, True), ('black_matte', holes, True)]
    cf = caliper_solid(.175, disc_t, calip_ang, .028)
    V, T = mesh_solid(lambda x, y, z: cf(x, y, z - zd), (-.23, -.23, zd - .07), (.23, .23, zd + .07), .004, tolerance=.0002)
    parts.append(('caliper', Mesh(V, T).with_normals(lambda x, y, z: cf(x, y, z - zd)), False))
    return parts
