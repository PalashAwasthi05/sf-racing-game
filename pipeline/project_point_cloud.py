"""Project an archived point cloud with bounded source and projection buffers."""
import zipfile

import numpy as np

from regen.bounded_dsm import CHUNK_ROWS, point_rows


def project_point_cloud(path, transformer, origin, chunk_rows=CHUNK_ROWS):
    """Return float32 local points, preserving all columns and source row order.

    Only the destination array scales with point count. Read the NPY header to
    allocate it, then stream source rows (including Fortran-order archives).
    Subtract the origin before casting, as in the eager projection.
    """
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    with zipfile.ZipFile(path) as archive, archive.open('points.npy') as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(source)
        else:
            raise ValueError(f'Unsupported points.npy version: {version}')
    if len(shape) != 2 or shape[1] < 4 or dtype.kind != 'f':
        raise ValueError('points must be a two-dimensional floating-point array with at least four columns')
    points = np.empty(shape, dtype=np.float32)
    cx, cy = origin
    start = 0
    with point_rows(path, chunk_rows) as chunks:
        for block in chunks:
            px, py = transformer.transform(block[:, 0], block[:, 1])
            local = points[start:start + len(block)]
            local[:, 0] = px - cx
            local[:, 1] = py - cy
            local[:, 2:] = block[:, 2:]
            start += len(block)
    return points
