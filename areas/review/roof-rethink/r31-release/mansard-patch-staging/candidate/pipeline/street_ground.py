"""Street surfaces stand on the LiDAR ground under the carriageway itself, not on the 3 m terrain grid.

The terrain grid takes the median of the 8 nearest ground returns at each 3 m node, so on a hill it climbs into the raised
lots, yards and steps beside a sidewalk: a kerbside sidewalk draped on it rose up to a metre above its own kerb, and terrain
triangles poked through the street. Here the carriageway is rasterised at 1 m and each cell takes the median of the ground
returns inside it (cells without returns, e.g. under parked cars, take the nearest measured cell). Every point within reach
of a carriageway takes the height of its nearest carriageway cell; past reach it eases back to the terrain, and the terrain
grid is held just under that surface so it never shows through a street. No location-specific input."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from shapely import contains_xy


def carriageway_heights(ground, road, lo, n, res=1., smooth_m=1.5, min_returns=2):
    """(height raster, distance-to-carriageway raster in metres) over n x n cells from lo at res; row = y, col = x."""
    xs = lo + np.arange(n) * res
    gx, gy = np.meshgrid(xs, xs)
    mask = contains_xy(road, gx.ravel(), gy.ravel()).reshape(n, n) if not road.is_empty else np.zeros((n, n), bool)
    ix = np.round((ground[:, 0] - lo) / res).astype(int); iy = np.round((ground[:, 1] - lo) / res).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < n) & (iy < n); ix, iy, z = ix[ok], iy[ok], ground[ok, 2]
    on = mask[iy, ix]; cell, z = (iy * n + ix)[on], z[on]
    measured = np.zeros(n * n, bool); h = np.zeros(n * n)
    if len(cell):
        order = np.lexsort((z, cell)); cell, z = cell[order], z[order]
        uniq, start, count = np.unique(cell, return_index=True, return_counts=True)
        keep = count >= min_returns; uniq, start, count = uniq[keep], start[keep], count[keep]
        h[uniq] = z[start + count // 2]; measured[uniq] = True
    measured = measured.reshape(n, n); h = h.reshape(n, n)
    if not measured.any():return None, None
    _, (ny, nx) = distance_transform_edt(~measured, return_indices=True)
    h = gaussian_filter(h[ny, nx], smooth_m / res)
    dist = distance_transform_edt(~mask) * res
    return h, dist


def build(ground, road, terrain, lo, step, n_grid, res=1., reach_m=7., ease_m=4., under_m=.05):
    """Return (elev, held terrain grid). elev(x, y) is the street height within reach_m of a carriageway, easing to the
    terrain over ease_m; the returned grid is the terrain lowered or raised to sit under_m beneath elev near streets."""
    size = (n_grid - 1) * step; n = int(round(size / res)) + 1
    h, dist = carriageway_heights(ground, road, lo, n, res)
    grid0 = terrain.copy()

    def terrain_elev(x, y):
        fx = np.clip((np.asarray(x) - lo) / step, 0, n_grid - 1.00001); fy = np.clip((np.asarray(y) - lo) / step, 0, n_grid - 1.00001)
        ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); fx = fx - ix; fy = fy - iy
        a, b, c, d = grid0[iy, ix], grid0[iy, ix + 1], grid0[iy + 1, ix], grid0[iy + 1, ix + 1]
        return np.where(fx + fy <= 1, a + (b - a) * fx + (c - a) * fy, d + (c - d) * (1 - fx) + (b - d) * (1 - fy))

    if h is None:return terrain_elev, grid0

    def sample(r, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        return map_coordinates(r, [np.clip((y.ravel() - lo) / res, 0, n - 1), np.clip((x.ravel() - lo) / res, 0, n - 1)], order=1).reshape(x.shape)

    def weight(x, y):
        return np.clip((sample(dist, x, y) - reach_m) / ease_m, 0., 1.)

    def elev(x, y):
        w = weight(x, y)
        return sample(h, x, y) * (1 - w) + terrain_elev(x, y) * w

    gx, gy = np.meshgrid(np.arange(n_grid) * step + lo, np.arange(n_grid) * step + lo)
    w = weight(gx, gy); near = w < 1
    held = grid0.copy(); held[near] = (elev(gx, gy) - under_m * (1 - w))[near]
    return elev, held
