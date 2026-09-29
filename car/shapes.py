"""Geometry helpers for the vehicle parts: 2D outlines, implicit primitives, swept tubes and parametric patches."""
import numpy as np
from mesher import Mesh, merge


# ---- 2D outlines ------------------------------------------------------------------------------------------------------
def catmull(points, n_per=12, closed=True):
    """A smooth curve through control points (centripetal-free uniform Catmull-Rom)."""
    P = np.asarray(points, float)
    if closed:
        P = np.concatenate([P[-1:], P, P[:2]])
    else:
        P = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return np.array(out)


def polygon_sdf(poly, u, v):
    """Signed distance (negative inside) from points (u, v) to a closed polygon."""
    P = np.asarray(poly, float)
    A, B = P, np.roll(P, -1, 0)
    shp = np.shape(u)
    q = np.stack([np.ravel(u), np.ravel(v)], 1)
    d2 = np.full(len(q), np.inf)
    inside = np.zeros(len(q), bool)
    for a, b in zip(A, B):
        e = b - a
        w = q - a
        t = np.clip((w @ e) / max(e @ e, 1e-18), 0, 1)
        dd = w - t[:, None] * e
        d2 = np.minimum(d2, (dd * dd).sum(1))
        c1 = (q[:, 1] >= a[1]) != (q[:, 1] >= b[1])
        xint = a[0] + (q[:, 1] - a[1]) * e[0] / np.where(e[1] == 0, 1e-18, e[1])
        inside ^= c1 & (q[:, 0] < xint)
    d = np.sqrt(d2)
    return np.where(inside, -d, d).reshape(shp)


def polyline_dist(line, u, v):
    """Unsigned distance from points to an open polyline."""
    P = np.asarray(line, float)
    shp = np.shape(u)
    q = np.stack([np.ravel(u), np.ravel(v)], 1)
    d2 = np.full(len(q), np.inf)
    for a, b in zip(P[:-1], P[1:]):
        e = b - a
        w = q - a
        t = np.clip((w @ e) / max(e @ e, 1e-18), 0, 1)
        dd = w - t[:, None] * e
        d2 = np.minimum(d2, (dd * dd).sum(1))
    return np.sqrt(d2).reshape(shp)


class Outline:
    """A closed 2D shape with a cached distance grid for fast evaluation on large point sets."""

    def __init__(self, poly, lo, hi, h=.002):
        self.poly = np.asarray(poly, float)
        self.lo, self.h = np.asarray(lo, float), h
        n = np.ceil((np.asarray(hi) - self.lo) / h).astype(int) + 1
        U, V = np.meshgrid(self.lo[0] + h * np.arange(n[0]), self.lo[1] + h * np.arange(n[1]), indexing='ij')
        self.grid = polygon_sdf(self.poly, U, V)
        self.n = n

    def __call__(self, u, v):
        fu = (np.asarray(u) - self.lo[0]) / self.h
        fv = (np.asarray(v) - self.lo[1]) / self.h
        iu = np.clip(np.floor(fu).astype(int), 0, self.n[0] - 2)
        iv = np.clip(np.floor(fv).astype(int), 0, self.n[1] - 2)
        a, b = np.clip(fu - iu, 0, 1), np.clip(fv - iv, 0, 1)
        g = self.grid
        d = (g[iu, iv] * (1 - a) * (1 - b) + g[iu + 1, iv] * a * (1 - b) + g[iu, iv + 1] * (1 - a) * b + g[iu + 1, iv + 1] * a * b)
        # outside the cached grid: fall back to a conservative distance to the grid box
        out = np.maximum(np.maximum(self.lo[0] - u, u - (self.lo[0] + self.h * (self.n[0] - 1))),
                         np.maximum(self.lo[1] - v, v - (self.lo[1] + self.h * (self.n[1] - 1))))
        return np.where(out > 0, np.maximum(d, out), d)


# ---- implicit primitives ------------------------------------------------------------------------------------------------
def box3(x, y, z, c, half, r):
    qx = np.abs(x - c[0]) - half[0] + r
    qy = np.abs(y - c[1]) - half[1] + r
    qz = np.abs(z - c[2]) - half[2] + r
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def capsule(x, y, z, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    e = b - a
    wx, wy, wz = x - a[0], y - a[1], z - a[2]
    t = np.clip((wx * e[0] + wy * e[1] + wz * e[2]) / (e @ e), 0, 1)
    dx, dy, dz = wx - t * e[0], wy - t * e[1], wz - t * e[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz) - r


def ellipsoid(x, y, z, c, r):
    """Approximate signed distance to an axis-aligned ellipsoid (good near the surface)."""
    px, py, pz = (x - c[0]) / r[0], (y - c[1]) / r[1], (z - c[2]) / r[2]
    k0 = np.sqrt(px * px + py * py + pz * pz)
    k1 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


# ---- explicit meshes ------------------------------------------------------------------------------------------------------
def frames(curve, up_hint):
    """Tangent and two normals along a polyline; `up_hint(i, point)` gives the preferred normal direction."""
    C = np.asarray(curve, float)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.array([up_hint(i, C[i]) for i in range(len(C))], float)
    N -= (N * T).sum(1)[:, None] * T
    N /= np.linalg.norm(N, axis=1)[:, None]
    B = np.cross(T, N)
    return T, N, B


def sweep(curve, profile, up_hint, closed_curve=False, caps=True):
    """Sweep a closed 2D profile (list of (b, n) offsets, counter-clockwise) along a 3D curve. The profile's first axis
    runs along B = T x N, its second along N. Returns a mesh with smooth normals across the profile."""
    C = np.asarray(curve, float)
    T, N, B = frames(C, up_hint)
    prof = np.asarray(profile, float)
    m = len(prof)
    # profile normals (2D, outward for counter-clockwise order)
    d = np.roll(prof, -1, 0) - np.roll(prof, 1, 0)
    pn = np.stack([d[:, 1], -d[:, 0]], 1)
    pn /= np.maximum(np.linalg.norm(pn, axis=1), 1e-12)[:, None]
    V = (C[:, None, :] + prof[None, :, 0, None] * B[:, None, :] + prof[None, :, 1, None] * N[:, None, :]).reshape(-1, 3)
    Nn = (pn[None, :, 0, None] * B[:, None, :] + pn[None, :, 1, None] * N[:, None, :]).reshape(-1, 3)
    tris = []
    rings = len(C)
    for i in range(rings - (0 if closed_curve else 1)):
        j = (i + 1) % rings
        a = i * m + np.arange(m)
        b = i * m + (np.arange(m) + 1) % m
        a2, b2 = j * m + np.arange(m), j * m + (np.arange(m) + 1) % m
        tris.append(np.stack([a, b2, b], 1))
        tris.append(np.stack([a, a2, b2], 1))
    mesh = Mesh(V, np.concatenate(tris), Nn)
    if caps and not closed_curve:
        parts = [mesh]
        for end, sgn in ((0, -1), (rings - 1, 1)):
            ring = C[end][None] + prof[:, 0, None] * B[end] + prof[:, 1, None] * N[end]
            c = ring.mean(0)
            Vc = np.concatenate([ring, c[None]])
            idx = np.arange(m)
            Tc = np.stack([idx, (idx + 1) % m, np.full(m, m)], 1)
            if sgn < 0:
                Tc = Tc[:, ::-1]
            # winding: normal along the curve direction at the far end, against it at the start
            nrm = np.repeat((T[end] * sgn)[None], m + 1, 0)
            cap = Mesh(Vc, Tc, nrm)
            fn = cap.face_normals()
            if (fn @ (T[end] * sgn)).mean() < 0:
                cap.T = cap.T[:, ::-1]
            parts.append(cap)
        mesh = merge(parts)
    return mesh


def rounded_rect(w, h, r, n=5, centre=(0., 0.)):
    """Counter-clockwise rounded rectangle profile (b across, n up)."""
    cx, cy = centre
    pts = []
    for (qx, qy, a0) in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for a in np.deg2rad(np.linspace(a0, a0 + 90, n)):
            pts.append((cx + qx + r * np.cos(a), cy + qy + r * np.sin(a)))
    return pts


def patch(fn, nu, nv, u_range=(-1, 1), v_range=(0, 1), both_sides=False):
    """Parametric surface patch: fn(U, V) -> (X, Y, Z) arrays. Normals from the parameter derivatives."""
    u = np.linspace(*u_range, nu)
    v = np.linspace(*v_range, nv)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    X, Y, Z = fn(U, Vv)
    P = np.stack([X, Y, Z], -1)
    du = np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    ln = np.linalg.norm(n, axis=-1)
    # a row or column that collapses to a point has no tangent plane: take the normal of the nearest valid neighbour
    bad = ln < 1e-9 * max(ln.max(), 1e-30)
    if bad.any():
        for j in range(nv):
            for i in np.nonzero(bad[:, j])[0]:
                for dj in (1, -1, 2, -2):
                    if 0 <= j + dj < nv and not bad[i, j + dj]:
                        n[i, j] = n[i, j + dj]
                        break
        ln = np.linalg.norm(n, axis=-1)
    n /= np.maximum(ln, 1e-12)[..., None]
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    T = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    m = Mesh(P.reshape(-1, 3), T, n.reshape(-1, 3))
    if both_sides:
        back = Mesh(m.V.copy(), m.T[:, ::-1].copy(), -m.N)
        m = merge([m, back])
    return m


def mirror_z(mesh):
    """The mesh reflected across z = 0 (winding flipped so it still faces out)."""
    return Mesh(mesh.V * np.array([1, 1, -1]), mesh.T[:, ::-1].copy(), None if mesh.N is None else mesh.N * np.array([1, 1, -1]))


def both_sides_z(mesh):
    return merge([mesh, mirror_z(mesh)])


def polyline_signed(line, u, v):
    """Distance to an open polyline, signed by the side of the nearest segment (left of its direction positive)."""
    P = np.asarray(line, float)
    shp = np.shape(u)
    q = np.stack([np.ravel(u), np.ravel(v)], 1)
    best = np.full(len(q), np.inf)
    sign = np.ones(len(q))
    for a, b in zip(P[:-1], P[1:]):
        e = b - a
        w = q - a
        t = np.clip((w @ e) / max(e @ e, 1e-18), 0, 1)
        dd = w - t[:, None] * e
        d2 = (dd * dd).sum(1)
        closer = d2 < best
        best = np.where(closer, d2, best)
        cr = e[0] * w[:, 1] - e[1] * w[:, 0]
        sign = np.where(closer, np.where(cr >= 0, 1., -1.), sign)
    return (np.sqrt(best) * sign).reshape(shp)
