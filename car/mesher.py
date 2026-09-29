"""Meshing of smooth implicit solids for the vehicle model.

A solid is a function f(x, y, z) -> signed distance (negative inside), evaluated on numpy arrays. `mesh_solid` samples
it on a regular grid in slabs (bounded memory), extracts the zero set by surface nets (one vertex per sign-changing
cell, one quad per sign-changing grid edge), moves every vertex onto the true surface with a few Newton steps and takes
the normal from the field's gradient, so shading follows the analytic surface rather than the grid. `split` cuts a mesh
exactly along the zero set of a second field (a paint boundary, a lamp outline, a panel gap), so material edges are
crisp curves instead of triangle staircases.
"""
import numpy as np


def grid_axes(lo, hi, h):
    """Sample positions covering [lo, hi] with spacing h (one extra sample of margin each side)."""
    return [np.arange(a - h, b + 1.5 * h, h) for a, b in zip(lo, hi)]


def sample(f, axes, slab=24):
    xs, ys, zs = axes
    out = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i0 in range(0, len(xs), slab):
        xi = xs[i0:i0 + slab]
        X = np.broadcast_to(xi[:, None, None], (len(xi),) + Y.shape)
        out[i0:i0 + len(xi)] = f(X, np.broadcast_to(Y, X.shape), np.broadcast_to(Z, X.shape))
    return out


_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(_CORNERS[a] - _CORNERS[b]).sum() == 1]


def surface_nets(F, axes):
    """Vertices (n,3) and triangles (m,3) of the zero set of samples F on the grid `axes`; triangles wind so that
    their normals point toward positive F (outside)."""
    xs, ys, zs = axes
    h = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
    inside = F < 0
    nx, ny, nz = F.shape
    cube = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for c, (i, j, k) in enumerate(_CORNERS):
        cube |= (inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k].astype(np.uint8) << c)
    active = (cube != 0) & (cube != 255)
    ci, cj, ck = np.nonzero(active)
    index = np.full(cube.shape, -1, np.int64)
    index[ci, cj, ck] = np.arange(len(ci))
    acc = np.zeros((len(ci), 3))
    cnt = np.zeros(len(ci))
    for a, b in _EDGES:
        pa, pb = _CORNERS[a], _CORNERS[b]
        fa = F[ci + pa[0], cj + pa[1], ck + pa[2]].astype(np.float64)
        fb = F[ci + pb[0], cj + pb[1], ck + pb[2]].astype(np.float64)
        cross = (fa < 0) != (fb < 0)
        t = np.where(cross, fa / np.where(cross, fa - fb, 1), 0)
        p = pa + t[:, None] * (pb - pa)
        acc[cross] += p[cross]
        cnt[cross] += 1
    local = acc / cnt[:, None]
    V = np.stack([xs[ci], ys[cj], zs[ck]], 1) + local * h
    quads = []
    # grid edges along each axis with a sign change; the four cells around the edge form a quad
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        a0 = inside[tuple(sl0)]
        a1 = inside[tuple(sl1)]
        change = a0 != a1
        # the edge needs cells on both sides in u and v: drop the first and last sample planes along u and v
        mask = np.zeros_like(change)
        s = [slice(None)] * 3
        s[u] = slice(1, -1)
        s[v] = slice(1, -1)
        mask[tuple(s)] = change[tuple(s)]
        e = np.nonzero(mask)
        e = np.stack(e, 1)
        if not len(e):
            continue
        cells = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = e.copy()
            c[:, u] += du
            c[:, v] += dv
            cells.append(index[c[:, 0], c[:, 1], c[:, 2]])
        q = np.stack(cells, 1)
        flip = ~a0[e[:, 0], e[:, 1], e[:, 2]]   # outside at the low end: the surface faces -axis
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    Q = np.concatenate(quads)
    Q = Q[(Q >= 0).all(1)]
    return V, Q


def triangulate(V, Q):
    """Split quads along their shorter diagonal."""
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    a = d02 <= d13
    T = np.concatenate([
        np.stack([Q[a, 0], Q[a, 1], Q[a, 2]], 1), np.stack([Q[a, 0], Q[a, 2], Q[a, 3]], 1),
        np.stack([Q[~a, 0], Q[~a, 1], Q[~a, 3]], 1), np.stack([Q[~a, 1], Q[~a, 2], Q[~a, 3]], 1)])
    return T


def gradient(f, P, eps=2e-4):
    g = np.empty_like(P)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        a, b = P + d, P - d
        g[:, k] = (f(a[:, 0], a[:, 1], a[:, 2]) - f(b[:, 0], b[:, 1], b[:, 2])) / (2 * eps)
    return g


def project(f, P, steps=4, limit=None):
    """Newton steps onto f = 0, each limited to `limit` (so a vertex never jumps to another sheet)."""
    P = P.copy()
    for _ in range(steps):
        v = f(P[:, 0], P[:, 1], P[:, 2])
        g = gradient(f, P)
        gg = np.maximum((g * g).sum(1), 1e-12)
        d = -(v / gg)[:, None] * g
        if limit is not None:
            n = np.linalg.norm(d, axis=1)
            d *= np.minimum(1, limit / np.maximum(n, 1e-12))[:, None]
        P += d
    return P


def normals(f, P):
    g = gradient(f, P)
    return g / np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]


def clean(V, T):
    """Drop degenerate triangles and unused vertices."""
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T = T[ok]
    used = np.unique(T)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[T]


def mesh_solid(f, lo, hi, h, steps=4, tolerance=None):
    """Mesh the solid f inside the box; with `tolerance` (metres) the watertight result is simplified by edge
    collapses that keep every point within that distance of the fine surface, and its vertices re-projected."""
    axes = grid_axes(lo, hi, h)
    F = sample(f, axes)
    V, Q = surface_nets(F, axes)
    del F
    T = triangulate(V, Q)
    V, T = clean(V, T)
    V = project(f, V, steps=steps, limit=h * .75)
    if tolerance:
        V, T = simplify(V, T, tolerance)
        V = project(f, V, steps=2, limit=h * .25)
    return V, T


def simplify(V, T, tolerance):
    import manifold3d as m3
    m = m3.Manifold(m3.Mesh(vert_properties=V.astype(np.float32), tri_verts=T.astype(np.uint32)))
    if m.status() != m3.Error.NoError:
        import sys
        print('simplify skipped: mesh is not a manifold (%s), %d triangles kept' % (m.status(), len(T)), file=sys.stderr)
        return V, T
    out = m.simplify(tolerance).to_mesh()
    return np.asarray(out.vert_properties, np.float64)[:, :3], np.asarray(out.tri_verts, np.int64)


class Mesh:
    """Triangle mesh with per-vertex normals."""

    def __init__(self, V, T, N=None):
        self.V, self.T = np.asarray(V, np.float64), np.asarray(T, np.int64)
        self.N = N

    def with_normals(self, f):
        self.N = normals(f, self.V)
        return self

    def face_normals(self):
        a, b, c = (self.V[self.T[:, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        return n / np.maximum(np.linalg.norm(n, axis=1), 1e-15)[:, None]

    def mirrored(self):
        """This mesh plus its mirror image across z = 0 (for a half built on z >= 0)."""
        V2 = self.V * np.array([1, 1, -1])
        T2 = self.T[:, ::-1] + len(self.V)
        N = None if self.N is None else np.concatenate([self.N, self.N * np.array([1, 1, -1])])
        return Mesh(np.concatenate([self.V, V2]), np.concatenate([self.T, T2]), N)


def split(mesh, g, f=None, steps=3):
    """Cut `mesh` along g = 0. Returns (inside, outside) meshes (g < 0, g >= 0). New vertices on cut edges are shared by
    both halves and, when the solid's field `f` is given, moved onto the curve where both fields vanish."""
    V, T = mesh.V, mesh.T
    gv = g(V[:, 0], V[:, 1], V[:, 2])
    neg = gv < 0
    s = neg[T].sum(1)
    keep_in = T[s == 3]
    keep_out = T[s == 0]
    mixed = T[(s == 1) | (s == 2)]
    if not len(mixed):
        return _sub(mesh, keep_in), _sub(mesh, keep_out)
    # rotate each mixed triangle so vertex 0 is the one alone on its side
    nm = neg[mixed]
    lone = np.where(nm.sum(1) == 1, np.argmax(nm, 1), np.argmin(nm, 1))
    r = np.stack([mixed[np.arange(len(mixed)), (lone + k) % 3] for k in range(3)], 1)
    a, b, c = r[:, 0], r[:, 1], r[:, 2]
    edges = np.concatenate([np.stack([a, b], 1), np.stack([a, c], 1)])
    key = np.sort(edges, 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    p0, p1 = V[uniq[:, 0]], V[uniq[:, 1]]
    g0, g1 = gv[uniq[:, 0]], gv[uniq[:, 1]]
    t = (g0 / (g0 - g1))[:, None]
    P = p0 + t * (p1 - p0)
    if f is not None:
        P_lin = P.copy()
        elen = np.linalg.norm(p1 - p0, axis=1)
        for _ in range(steps):
            # alternate: onto g = 0 along its gradient, then back onto the solid's surface; each step is bounded so a
            # vertex never wanders off its edge (a far jump folds the surface into a visible shard)
            for fn in (g, f):
                v = fn(P[:, 0], P[:, 1], P[:, 2])
                gr = gradient(fn, P)
                d = -(v / np.maximum((gr * gr).sum(1), 1e-12))[:, None] * gr
                n = np.linalg.norm(d, axis=1)
                d *= np.minimum(1, .003 / np.maximum(n, 1e-12))[:, None]
                P = P + d
        off = np.linalg.norm(P - P_lin, axis=1) > .25 * elen + 1e-4
        if off.any():
            # keep the edge's own crossing, only settled onto the surface by a short bounded step (a free Newton step
            # from a point inside a thin wall can land on the far skin)
            Q = P_lin[off]
            lim = .1 * elen[off]
            for _ in range(3):
                v = f(Q[:, 0], Q[:, 1], Q[:, 2])
                gr = gradient(f, Q)
                d = -(v / np.maximum((gr * gr).sum(1), 1e-12))[:, None] * gr
                n = np.linalg.norm(d, axis=1)
                Q = Q + d * np.minimum(1, lim / np.maximum(n, 1e-12))[:, None]
            P[off] = Q
    base = len(V)
    n_ab = base + inv[:len(r)]
    n_ac = base + inv[len(r):]
    lone_neg = neg[a]
    t_lone = np.stack([a, n_ab, n_ac], 1)
    t_q1 = np.stack([n_ab, b, c], 1)
    t_q2 = np.stack([n_ab, c, n_ac], 1)
    V2 = np.concatenate([V, P])
    N2 = None
    if mesh.N is not None:
        if f is not None:
            N2 = np.concatenate([mesh.N, normals(f, P)])
        else:
            n = mesh.N[uniq[:, 0]] + t * (mesh.N[uniq[:, 1]] - mesh.N[uniq[:, 0]])
            N2 = np.concatenate([mesh.N, n / np.linalg.norm(n, axis=1)[:, None]])
    ins = np.concatenate([keep_in, t_lone[lone_neg], t_q1[~lone_neg], t_q2[~lone_neg]])
    out = np.concatenate([keep_out, t_lone[~lone_neg], t_q1[lone_neg], t_q2[lone_neg]])
    whole = Mesh(V2, T, N2)
    return _sub(whole, ins), _sub(whole, out)


def _sub(mesh, T):
    if not len(T):
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64), None if mesh.N is None else np.zeros((0, 3)))
    used = np.unique(T)
    remap = np.full(len(mesh.V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.V[used], remap[T], None if mesh.N is None else mesh.N[used])


def merge(meshes):
    meshes = [m for m in meshes if len(m.T)]
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)))
    off = np.cumsum([0] + [len(m.V) for m in meshes[:-1]])
    V = np.concatenate([m.V for m in meshes])
    T = np.concatenate([m.T + o for m, o in zip(meshes, off)])
    N = None if any(m.N is None for m in meshes) else np.concatenate([m.N for m in meshes])
    return Mesh(V, T, N)


def refine_near(mesh, fields, f, h_target=.006, rounds=5, lines=(), h_line=.004):
    """Split triangles that lie within their own size of any field's zero set until they are smaller than h_target,
    with conforming closure (neighbours of a split edge are split too, so no cracks), moving new vertices onto the
    solid's surface f. A thin region (a reflector strip, a panel gap) can then be cut exactly even where the
    simplified mesh had large flat triangles."""
    V, T, N = mesh.V, mesh.T, mesh.N
    # barycentric sample grid (15 points): a region thinner than a triangle still shows as a sign change
    bary = np.array([(i, j, 4 - i - j) for i in range(5) for j in range(5 - i)], float) / 4
    for _ in range(rounds):
        P = V[T]
        size = np.max(np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2), axis=1)
        big = size > h_target
        near = np.zeros(len(T), bool)
        if big.any():
            S = np.einsum('kj,tjd->tkd', bary, P[big]).reshape(-1, 3)
            hit = np.zeros(big.sum(), bool)
            corners = [int(np.nonzero(bary[:, j] == 1)[0][0]) for j in range(3)]   # sample index of each vertex
            for g in fields:
                v = g(S[:, 0], S[:, 1], S[:, 2]).reshape(-1, len(bary))
                # a straight cut between the corner values is exact enough unless the samples disagree with it
                # (a band thinner than the triangle, a boundary curving inside it)
                lin = np.einsum('kj,tj->tk', bary[:, [0, 1, 2]], v[:, corners])
                hit |= ((v < 0) != (lin < 0)).any(1)
            near[np.nonzero(big)[0]] = hit
        # thin bands round centre lines (panel gaps): refine wherever a line passes within a triangle's size
        if len(lines):
            cen = P.mean(1)
            thin = np.zeros(len(T), bool)
            for g in lines:
                thin |= np.abs(g(cen[:, 0], cen[:, 1], cen[:, 2])) < size * .8
            near |= thin & (size > h_line)
        if not near.any():
            break
        e = np.sort(np.concatenate([T[near][:, [0, 1]], T[near][:, [1, 2]], T[near][:, [2, 0]]]), 1)
        marked = np.unique(e, axis=0)
        mids = project(f, (V[marked[:, 0]] + V[marked[:, 1]]) / 2, steps=3, limit=h_target)
        base = len(V)
        V = np.concatenate([V, mids])
        if N is not None:
            N = np.concatenate([N, normals(f, mids)])
        # edge -> midpoint index lookup
        key = marked[:, 0] * (base + 1) + marked[:, 1]
        order = np.argsort(key)
        skey = key[order]

        def mid(a, b):
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            k = lo * (base + 1) + hi
            i = np.searchsorted(skey, k)
            i = np.minimum(i, len(skey) - 1)
            hit = skey[i] == k
            return np.where(hit, base + order[i], -1)
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
        n = (mab >= 0).astype(int) + (mbc >= 0) + (mca >= 0)
        out = [T[n == 0]]
        # one split edge: rotate so it is (a, b)
        for sel, (p, q, r, m) in ((n == 1) & (mab >= 0), (a, b, c, mab)), ((n == 1) & (mbc >= 0), (b, c, a, mbc)), ((n == 1) & (mca >= 0), (c, a, b, mca)):
            out += [np.stack([p[sel], m[sel], r[sel]], 1), np.stack([m[sel], q[sel], r[sel]], 1)]
        # two split edges: rotate so the unsplit one is (c, a)
        for sel, (p, q, r, m1, m2) in (((n == 2) & (mca < 0), (a, b, c, mab, mbc)), ((n == 2) & (mab < 0), (b, c, a, mbc, mca)),
                                       ((n == 2) & (mbc < 0), (c, a, b, mca, mab))):
            out += [np.stack([m1[sel], q[sel], m2[sel]], 1), np.stack([p[sel], m1[sel], m2[sel]], 1), np.stack([p[sel], m2[sel], r[sel]], 1)]
        s3 = n == 3
        out += [np.stack([a[s3], mab[s3], mca[s3]], 1), np.stack([mab[s3], b[s3], mbc[s3]], 1),
                np.stack([mca[s3], mbc[s3], c[s3]], 1), np.stack([mab[s3], mbc[s3], mca[s3]], 1)]
        T = np.concatenate(out)
    return Mesh(V, T, N)


def ray_hit(f, P, d, far=1.0, iters=60):
    """First point where f < 0 along rays from P in direction d (sphere tracing, then bisection)."""
    P = np.atleast_2d(np.asarray(P, float))
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    t = np.zeros(len(P))
    for _ in range(iters):
        Q = P + t[:, None] * d
        v = f(Q[:, 0], Q[:, 1], Q[:, 2])
        t = np.where(v > 1e-5, t + np.maximum(v * .8, 1e-4), t)
        t = np.minimum(t, far)
    return P + t[:, None] * d


def dual_contour(F, axes, f, reg=.01):
    """Like surface_nets, but each cell's vertex minimises the distance to the tangent planes at its edge crossings
    (normals from the field f), so edges and corners smaller than a cell come out sharp instead of rounded into a
    sawtooth. Vertices whose solution leaves the cell fall back to the mean crossing."""
    xs, ys, zs = axes
    h = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
    inside = F < 0
    nx, ny, nz = F.shape
    cube = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for c, (i, j, k) in enumerate(_CORNERS):
        cube |= (inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k].astype(np.uint8) << c)
    active = (cube != 0) & (cube != 255)
    del cube
    ci, cj, ck = np.nonzero(active)
    del active
    origin = np.stack([xs[ci], ys[cj], zs[ck]], 1)
    n = len(ci)
    A = np.zeros((n, 3, 3))
    b = np.zeros((n, 3))
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    crossings, owners = [], []
    for a_, b_ in _EDGES:
        pa, pb = _CORNERS[a_], _CORNERS[b_]
        fa = F[ci + pa[0], cj + pa[1], ck + pa[2]].astype(np.float64)
        fb = F[ci + pb[0], cj + pb[1], ck + pb[2]].astype(np.float64)
        cross = (fa < 0) != (fb < 0)
        idx = np.nonzero(cross)[0]
        # false position on the true field along the edge: the grid's linear estimate is off near features
        A_ = origin[idx] + pa * h
        B_ = origin[idx] + pb * h
        va, vb = fa[idx], fb[idx]
        for _ in range(4):
            t = va / (va - vb)
            p = A_ + t[:, None] * (B_ - A_)
            vp = f(p[:, 0], p[:, 1], p[:, 2])
            same = (vp < 0) == (va < 0)
            A_ = np.where(same[:, None], p, A_)
            va = np.where(same, vp, va)
            B_ = np.where(same[:, None], B_, p)
            vb = np.where(same, vb, vp)
        t = va / (va - vb)
        p = A_ + t[:, None] * (B_ - A_)
        crossings.append(p)
        owners.append(idx)
    P = np.concatenate(crossings)
    own = np.concatenate(owners)
    N = normals(f, P)
    np.add.at(acc, own, P)
    np.add.at(cnt, own, 1)
    mass = acc / cnt[:, None]
    np.add.at(A, own, N[:, :, None] * N[:, None, :])
    np.add.at(b, own, N * (N * P).sum(1)[:, None])
    # regularise toward the mass point (flat regions have a rank-deficient system)
    lam = reg * np.maximum(cnt, 1)
    A += lam[:, None, None] * np.eye(3)
    b += lam[:, None] * mass
    X = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    lo, hi = origin - .25 * h, origin + 1.25 * h
    bad = ((X < lo) | (X > hi)).any(1)
    X[bad] = mass[bad]
    index = np.full((nx - 1, ny - 1, nz - 1), -1, np.int64)
    index[ci, cj, ck] = np.arange(n)
    quads = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        a0 = inside[tuple(sl0)]
        a1 = inside[tuple(sl1)]
        change = a0 != a1
        mask = np.zeros_like(change)
        s = [slice(None)] * 3
        s[u] = slice(1, -1)
        s[v] = slice(1, -1)
        mask[tuple(s)] = change[tuple(s)]
        e = np.stack(np.nonzero(mask), 1)
        if not len(e):
            continue
        cells = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = e.copy()
            c[:, u] += du
            c[:, v] += dv
            cells.append(index[c[:, 0], c[:, 1], c[:, 2]])
        q = np.stack(cells, 1)
        flip = ~a0[e[:, 0], e[:, 1], e[:, 2]]
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    Q = np.concatenate(quads)
    Q = Q[(Q >= 0).all(1)]
    return X, Q


def mesh_solid_sharp(f, lo, hi, h, tolerance=None):
    """mesh_solid with dual contouring: sharp creases stay sharp."""
    axes = grid_axes(lo, hi, h)
    F = sample(f, axes)
    V, Q = dual_contour(F, axes, f)
    del F
    T = triangulate(V, Q)
    V, T = clean(V, T)
    if tolerance:
        V, T = fix_nonmanifold(V, T)
        V, T = simplify(V, T, tolerance)
    return V, T


def autosmooth(mesh, f=None, angle=35.):
    """Vertex normals that stay smooth across gentle curvature but split at creases sharper than `angle`: each corner
    takes the area-weighted mean of the face normals round its vertex that lie within the angle of its own face.
    Where the field f is given, smooth corners use its gradient instead (exact on curved surfaces)."""
    V, T = mesh.V, mesh.T
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b - a, c - a)
    area = np.linalg.norm(fn, axis=1)
    keep = area > 1e-14                     # zero-area slivers draw nothing and have no normal
    T = T[keep]
    fn, area = fn[keep], area[keep]
    fu = fn / area[:, None]
    # slivers (from cutting along curves) have unreliable face normals: they take no part in finding creases
    e2 = sum(np.sum((V[T[:, i]] - V[T[:, (i + 1) % 3]]) ** 2, 1) for i in range(3))
    sliver = (2 * np.sqrt(3) * area / np.maximum(e2, 1e-30)) < .08
    corners = T.reshape(-1)
    face_of = np.repeat(np.arange(len(T)), 3)
    order = np.argsort(corners, kind='stable')
    cs, fs = corners[order], face_of[order]
    starts = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
    ends = np.r_[starts[1:], len(cs)]
    cos_t = np.cos(np.deg2rad(angle))
    out_n = np.zeros((len(corners), 3))
    creased = np.zeros(len(corners), bool)
    # vertices by valence: process groups of equal size together
    val = ends - starts
    for k in np.unique(val):
        g = np.nonzero(val == k)[0]
        idx = starts[g][:, None] + np.arange(k)[None, :]      # (m, k) positions in the sorted list
        F_ = fs[idx]                                           # faces round each vertex
        Nf = fu[F_]                                            # (m, k, 3)
        W = area[F_]
        dots = np.einsum('mik,mjk->mij', Nf, Nf)               # (m, k, k)
        sl = sliver[F_]                                        # (m, k)
        close = (dots >= cos_t) | sl[:, None, :] | sl[:, :, None]
        acc = np.einsum('mij,mj,mjd->mid', close.astype(float), W, Nf)
        acc /= np.maximum(np.linalg.norm(acc, axis=2), 1e-20)[:, :, None]
        out_n[order[idx]] = acc
        creased[order[idx]] = ~close.all(axis=(1, 2))[:, None].repeat(k, 1)
    if f is not None:
        smooth = ~creased
        vi = corners[smooth]
        g = normals(f, V[vi])
        ok = np.linalg.norm(g, axis=1) > .5
        sel = np.nonzero(smooth)[0][ok]
        out_n[sel] = g[ok]
    # any corner still without a normal takes its own face's
    weak = np.linalg.norm(out_n, axis=1) < .5
    out_n[weak] = fu[face_of[weak]]
    # unique (vertex, normal) pairs become the new vertices
    key = np.concatenate([corners[:, None].astype(np.float64), np.round(out_n * 1e4)], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    V2 = V[uk[:, 0].astype(np.int64)]
    N2 = np.zeros((len(uk), 3))
    N2[inv] = out_n
    N2 /= np.maximum(np.linalg.norm(N2, axis=1), 1e-20)[:, None]
    return Mesh(V2, inv.reshape(-1, 3), N2)


def fix_nonmanifold(V, T):
    """Where two sheets of a thin wall touch inside one grid cell, an edge ends up shared by four faces. Each vertex on
    such an edge is split into one copy per fan of faces around it, so every sheet keeps its own edge (the sheets still
    touch in space, which is harmless for drawing)."""
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    bad_v = np.unique(u[c > 2])
    if not len(bad_v):
        return V, T
    V = [v for v in V]
    T = T.copy()
    # vertex -> incident triangles, once (a scan of all triangles per vertex took minutes on a busy mesh)
    corners = T.reshape(-1)
    order = np.argsort(corners, kind='stable')
    cs = corners[order]
    lo = np.searchsorted(cs, bad_v, 'left')
    hi = np.searchsorted(cs, bad_v, 'right')
    tri_of = {int(v): list(np.unique(order[a:b] // 3)) for v, a, b in zip(bad_v, lo, hi)}
    for vid, tris in tri_of.items():
        # fans: triangles round vid connected through an edge (vid, w)
        parent = {t: t for t in tris}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        by_edge = {}
        for t in tris:
            for w in T[t]:
                if w != vid:
                    by_edge.setdefault(int(w), []).append(t)
        for w, ts in by_edge.items():
            if len(ts) == 2:                      # a manifold edge joins exactly two faces of one fan
                a, b = find(ts[0]), find(ts[1])
                parent[a] = b
        roots = sorted({find(t) for t in tris})
        for r in roots[1:]:
            V.append(V[vid])
            nid = len(V) - 1
            for t in tris:
                if find(t) == r:
                    T[t][T[t] == vid] = nid
    return np.array(V), T
