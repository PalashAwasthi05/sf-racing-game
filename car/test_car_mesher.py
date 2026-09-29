import numpy as np
from mesher import mesh_solid, Mesh, split, merge, refine_near, simplify, ray_hit, mesh_solid_sharp, autosmooth


def sphere(x, y, z):
    return np.sqrt(x * x + y * y + z * z) - 0.5


def edge_use(T):
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return c


def test_sphere_closed_outward_accurate():
    V, T = mesh_solid(sphere, (-.6, -.6, -.6), (.6, .6, .6), .03)
    m = Mesh(V, T).with_normals(sphere)
    assert (edge_use(T) == 2).all()                       # watertight: every edge shared by two triangles
    r = np.linalg.norm(V, axis=1)
    assert np.abs(r - .5).max() < 1e-4                    # vertices on the true surface
    fn = m.face_normals()
    c = V[T].mean(1)
    assert ((fn * c).sum(1) > 0).mean() > .999            # triangles wind outward
    vol = np.abs(np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]]))).sum() / 6
    assert abs(vol - 4 / 3 * np.pi * .125) / (4 / 3 * np.pi * .125) < .01


def test_split_is_exact_and_partitions():
    V, T = mesh_solid(sphere, (-.6, -.6, -.6), (.6, .6, .6), .03)
    m = Mesh(V, T).with_normals(sphere)
    plane = lambda x, y, z: y - .1
    a, b = split(m, plane, sphere)
    ya = a.V[:, 1]
    assert ya.max() < .1 + 1e-6 and b.V[:, 1].min() > .1 - 1e-6
    # the cut curve lies on both the sphere and the plane
    cut = a.V[np.abs(a.V[:, 1] - .1) < 1e-6]
    assert len(cut) > 50 and np.abs(np.linalg.norm(cut, axis=1) - .5).max() < 1e-4
    # areas add up to the whole
    area = lambda mm: np.linalg.norm(np.cross(mm.V[mm.T[:, 1]] - mm.V[mm.T[:, 0]], mm.V[mm.T[:, 2]] - mm.V[mm.T[:, 0]]), axis=1).sum() / 2
    assert abs(area(a) + area(b) - area(m)) / area(m) < 1e-3
    w = merge([a, b])
    assert len(w.T) == len(a.T) + len(b.T)


def test_refine_then_split_catches_a_thin_band_on_a_coarse_mesh():
    V, T = mesh_solid(sphere, (-.6, -.6, -.6), (.6, .6, .6), .03)
    V, T = simplify(V, T, .004)                      # large flat triangles
    m = Mesh(V, T).with_normals(sphere)
    band = lambda x, y, z: np.abs(y - .1) - .006     # 12 mm band, much thinner than the triangles
    r = refine_near(m, [band], sphere, h_target=.005)
    assert (edge_use(r.T) == 2).all()                # still watertight: conforming splits leave no cracks
    inside, _ = split(r, band, sphere)
    area = np.linalg.norm(np.cross(inside.V[inside.T[:, 1]] - inside.V[inside.T[:, 0]], inside.V[inside.T[:, 2]] - inside.V[inside.T[:, 0]]), axis=1).sum() / 2
    expected = 2 * np.pi * .5 * .012                 # zone of a sphere: 2 pi R h
    assert abs(area - expected) / expected < .05


def test_ray_hit_lands_on_the_surface_along_the_ray():
    p = ray_hit(sphere, [[2, .1, .2]], (-1, 0, 0), far=3)[0]
    assert abs(np.linalg.norm(p) - .5) < 1e-4 and abs(p[1] - .1) < 1e-9 and abs(p[2] - .2) < 1e-9


def cube(x, y, z):
    q = np.stack([np.abs(x), np.abs(y), np.abs(z)]) - .3
    return np.linalg.norm(np.maximum(q, 0), axis=0) + np.minimum(q.max(0), 0)


def test_dual_contouring_keeps_edges_sharp():
    V, T = mesh_solid_sharp(cube, (-.4, -.4, -.4), (.4, .4, .4), .037)   # grid not aligned with the faces
    assert (edge_use(T) == 2).all()
    # every vertex lies on the cube's surface, including those on its edges and corners
    assert np.abs(cube(V[:, 0], V[:, 1], V[:, 2])).max() < 2e-3
    assert np.abs(V).max() > .299                                          # corners reached, not rounded off
    m = autosmooth(Mesh(V, T))
    fn = m.face_normals()
    # split normals: every corner normal matches its face on a cube (all creases are 90 degrees)
    cn = m.N[m.T]
    assert (np.einsum('tkd,td->tk', cn, fn) > .99).mean() > .98
