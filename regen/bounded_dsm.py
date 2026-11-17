"""Exact facade DSM construction with bounded point and projection buffers."""
from contextlib import contextmanager
import shutil
import tempfile
import zipfile

import numpy as np
from pyproj import Transformer
from shapely import contains_xy

CHUNK_ROWS = 100_000


@contextmanager
def point_rows(path, chunk_rows=CHUNK_ROWS):
    """Yield a row iterator without np.load's eager compressed-member allocation.

    C-order arrays stream directly. Unusual Fortran-order archives are expanded to
    a temporary disk file, then read by column into bounded row buffers.
    """
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    with zipfile.ZipFile(path) as archive, archive.open('points.npy') as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f'Unsupported points.npy version: {version}')
        if len(shape) != 2 or shape[1] < 4 or dtype.kind != 'f':
            raise ValueError('points must be a two-dimensional floating-point array with at least four columns')
        def read_array(stream, count):
            raw = stream.read(count * dtype.itemsize)
            if len(raw) != count * dtype.itemsize:
                raise ValueError('Truncated points.npy')
            return np.frombuffer(raw, dtype=dtype)
        if not fortran:
            def rows():
                for start in range(0, shape[0], chunk_rows):
                    n = min(chunk_rows, shape[0] - start)
                    yield read_array(source, n * shape[1]).reshape(n, shape[1])
            yield rows()
        else:
            with tempfile.TemporaryFile() as disk:
                shutil.copyfileobj(source, disk, length=1024 * 1024)
                def rows():
                    for start in range(0, shape[0], chunk_rows):
                        n = min(chunk_rows, shape[0] - start)
                        block = np.empty((n, shape[1]), dtype=dtype)
                        for col in range(shape[1]):
                            disk.seek((col * shape[0] + start) * dtype.itemsize)
                            block[:, col] = read_array(disk, n)
                        yield block
                yield rows()


def disk_median(stream, count, chunk_rows=CHUNK_ROWS):
    """Exact float64 median; eight byte-radix passes, bounded RAM even on flat terrain."""
    if not count:
        return float('nan')
    if count <= chunk_rows:
        stream.seek(0)
        return float(np.median(np.frombuffer(stream.read(count * 8), dtype=np.float64)))
    ranks = [(count - 1) // 2, count // 2]
    prefixes = [0, 0]
    sign = np.uint64(1 << 63)
    for shift in range(56, -1, -8):
        hist = np.zeros((2, 256), dtype=np.int64)
        stream.seek(0)
        for start in range(0, count, chunk_rows):
            vals = np.frombuffer(stream.read(min(chunk_rows, count - start) * 8), dtype=np.float64)
            if np.isnan(vals).any():
                return float('nan')
            bits = vals.view(np.uint64)
            keys = np.where(bits & sign, ~bits, bits ^ sign)
            for i in range(2):
                selected = keys if shift == 56 else keys[(keys >> np.uint64(shift + 8)) == prefixes[i]]
                digits = ((selected >> np.uint64(shift)) & np.uint64(255)).astype(np.int64)
                hist[i] += np.bincount(digits, minlength=256)
        for i in range(2):
            cumulative = np.cumsum(hist[i])
            digit = int(np.searchsorted(cumulative, ranks[i], side='right'))
            ranks[i] -= int(cumulative[digit - 1]) if digit else 0
            prefixes[i] = (prefixes[i] << 8) | digit
    keys = np.asarray(prefixes, dtype=np.uint64)
    bits = np.where(keys & sign, keys ^ sign, ~keys)
    return float(np.mean(bits.view(np.float64)))


def build_dsm(path, frame, footprint, cell=.5, chunk_rows=CHUNK_ROWS):
    """Keep class 1 in supplied buffered footprints; ground datum is class 2 within 40m."""
    projection = Transformer.from_crs(3857, frame.ll.target_crs, always_xy=True)
    def projected(points):
        x, y = projection.transform(points[:, 0], points[:, 1])
        x -= frame.cx
        y -= frame.cy
        keep = points[:, 3] == 1
        keep[keep] = contains_xy(footprint, x[keep], y[keep])
        return x, y, keep

    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    ground_count = 0
    with tempfile.TemporaryFile() as ground:
        with point_rows(path, chunk_rows) as chunks:
            for points in chunks:
                x, y, keep = projected(points)
                heights = np.asarray(points[(points[:, 3] == 2) & (np.hypot(x, y) < 40), 2], dtype=np.float64)
                ground.write(heights.tobytes())
                ground_count += len(heights)
                if keep.any():
                    x0, y0 = min(x0, x[keep].min()), min(y0, y[keep].min())
                    x1, y1 = max(x1, x[keep].max()), max(y1, y[keep].max())
        base = disk_median(ground, ground_count, chunk_rows)
    if not np.isfinite(x0) or not np.isfinite(y0):
        raise ValueError('No class-1 points inside buffered footprints')
    grid = np.full((int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1), -np.inf, np.float32)
    with point_rows(path, chunk_rows) as chunks:
        for points in chunks:
            x, y, keep = projected(points)
            ix = ((x[keep] - x0) / cell).astype(int)
            iy = ((y[keep] - y0) / cell).astype(int)
            z = (points[keep, 2] - base).astype(np.float32)
            np.maximum.at(grid, (ix, iy), z)
    return grid, x0, y0
