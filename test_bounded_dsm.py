"""Small synthetic equivalence checks; never opens area point archives."""
from pathlib import Path
from tempfile import TemporaryFile
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import box

from regen.bounded_dsm import build_dsm, disk_median, point_rows


@pytest.mark.parametrize('fortran', [False, True])
@pytest.mark.parametrize('chunk', [1, 7, 1000])
def test_dsm_matches_eager(tmp_path, fortran, chunk):
    # A different target CRS catches accidental reinstatement of a geographic constant.
    ll = Transformer.from_crs(4326, 32632, always_xy=True)
    cx, cy = ll.transform(9., 48.)
    frame = SimpleNamespace(ll=ll, cx=cx, cy=cy)
    rng = np.random.default_rng(23)
    local = rng.uniform(-20, 20, (301, 2))
    local[:4] = [[0, 0], [0, 0], [10.2, 0], [10.4, 0]]
    web = Transformer.from_crs(ll.target_crs, 3857, always_xy=True)
    wx, wy = web.transform(local[:, 0] + cx, local[:, 1] + cy)
    points = np.column_stack((wx, wy, rng.uniform(100, 150, 301), rng.choice([1, 2, 6], 301), np.zeros(301)))
    points[:4, 3] = 1
    points[4:8, 3] = 2
    # Ground outside the 40m datum radius must not affect the median.
    points[-1, :2] = web.transform(cx + 80, cy)
    points[-1, 2:4] = [-10000, 2]
    points = np.asfortranarray(points) if fortran else points
    path = tmp_path / 'points.npz'
    np.savez_compressed(path, points=points)
    footprint = box(-10, -10, 10, 10).buffer(.3)
    actual, ax, ay = build_dsm(path, frame, footprint, chunk_rows=chunk)

    projection = Transformer.from_crs(3857, ll.target_crs, always_xy=True)
    x, y = projection.transform(points[:, 0], points[:, 1])
    x, y = x - cx, y - cy
    z, c = points[:, 2], points[:, 3]
    base = float(np.median(z[(c == 2) & (np.hypot(x, y) < 40)]))
    z = z - base
    keep = c == 1
    keep[keep] = contains_xy(footprint, x[keep], y[keep])
    x, y, z = x[keep], y[keep], z[keep]
    x0, y0 = x.min(), y.min()
    ix, iy = ((x - x0) / .5).astype(int), ((y - y0) / .5).astype(int)
    expected = np.full((ix.max() + 1, iy.max() + 1), -np.inf, np.float32)
    np.maximum.at(expected, (ix, iy), z.astype(np.float32))
    assert (ax, ay) == (x0, y0)
    np.testing.assert_array_equal(actual, expected)
    with point_rows(path, chunk) as blocks:
        result = []
        for block in blocks:
            assert len(block) <= chunk
            result.append(block.copy())
    np.testing.assert_array_equal(np.concatenate(result), points)


@pytest.mark.parametrize('values', [[], [1], [-9, 2], [5, -8, 3],
    [4, 4, 4, -5, 2, 3], [-np.inf, 1, 2, np.inf], [1, np.nan, 3],
    [-0., 0., -1., 1.], list(np.random.default_rng(42).normal(size=101))])
def test_external_median_exact(values):
    values = np.asarray(values, dtype=np.float64)
    with TemporaryFile() as stream:
        stream.write(values.tobytes())
        result = disk_median(stream, len(values), chunk_rows=1)
    expected = np.median(values) if len(values) else np.nan
    np.testing.assert_equal(result, expected)


def test_missing_ground_preserves_nan_dsm(tmp_path):
    frame = SimpleNamespace(ll=Transformer.from_crs(4326, 3857, always_xy=True), cx=0., cy=0.)
    path = tmp_path / 'points.npz'
    np.savez_compressed(path, points=np.array([[0., 0., 5., 1.], [1., 1., 8., 1.]]))
    grid, _, _ = build_dsm(path, frame, box(-2, -2, 2, 2), chunk_rows=1)
    assert np.isnan(grid[0, 0]) and np.isnan(grid[2, 2])
    assert np.isneginf(grid[0, 1])
