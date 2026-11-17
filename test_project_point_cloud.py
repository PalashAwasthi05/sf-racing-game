"""Synthetic exact-equivalence checks; never loads an area cloud."""
from io import BytesIO
import zipfile

import numpy as np
import pytest
from pyproj import Transformer

from pipeline.project_point_cloud import project_point_cloud


@pytest.mark.parametrize('fortran', [False, True])
@pytest.mark.parametrize('chunk_rows', [1, 7, 1000])
@pytest.mark.parametrize('version', [(1, 0), (2, 0)])
def test_projection_matches_eager(tmp_path, monkeypatch, fortran, chunk_rows, version):
    rng = np.random.default_rng(42)
    source = rng.normal(size=(103, 5))
    source[:, :2] = source[:, :2] * 100 + [1_000_000, 6_000_000]
    source[:, 3] = rng.choice([1, 2, 6], len(source))
    source[0, 2:] = [np.nan, 2, -0.]
    source = np.asfortranarray(source) if fortran else source
    path = tmp_path / 'points.npz'
    member = BytesIO()
    np.lib.format.write_array(member, source, version=version)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('points.npy', member.getvalue())
    transformer = Transformer.from_crs(3857, 32632, always_xy=True)
    origin = transformer.transform(1_000_000, 6_000_000)
    with np.load(path) as archive:
        eager = archive['points']
    px, py = transformer.transform(eager[:, 0], eager[:, 1])
    expected = np.empty(eager.shape, dtype=np.float32)
    expected[:, 0] = px - origin[0]
    expected[:, 1] = py - origin[1]
    expected[:, 2:] = eager[:, 2:]

    def no_eager_load(*args, **kwargs):
        raise AssertionError('Source archive must be streamed')

    monkeypatch.setattr(np, 'load', no_eager_load)

    class BoundedTransformer:
        calls = 0

        def transform(self, x, y):
            assert len(x) <= chunk_rows
            self.calls += 1
            return transformer.transform(x, y)

    bounded = BoundedTransformer()
    actual = project_point_cloud(path, bounded, origin, chunk_rows)
    assert bounded.calls == (len(source) + chunk_rows - 1) // chunk_rows
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_empty_cloud(tmp_path):
    path = tmp_path / 'points.npz'
    np.savez_compressed(path, points=np.empty((0, 5)))
    actual = project_point_cloud(path, None, (0., 0.))
    assert actual.shape == (0, 5)
    assert actual.dtype == np.float32


def test_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match='chunk_rows must be positive'):
        project_point_cloud(tmp_path / 'unused.npz', None, (0., 0.), 0)
