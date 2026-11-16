"""Rasterize actual street-facing R31 triangles in metric orthographic wall charts.

All geometry and chart arrays are east, north, up in metres. Chart pixel centres
are on the sampled triangle surfaces. The chart depth is the first intersection
seen looking out from the street, and the original R31 vertices remain available
for rendering the overlay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from regen.glb_stream import GltfStream
from regen.host_shell_apertures import GLB_TO_SOURCE


def raster(xy, depths, shape):
    """Exact barycentric single-sample depth and triangle ID at each pixel centre.

    For perspective callers pass -1/cameraDepth to interpolate inverse depth.
    Each chunk enumerates clipped triangle bounding boxes with NumPy.
    """
    height, width = shape
    xy = np.asarray(xy, np.float32)
    depths = np.asarray(depths, np.float32)
    low = np.maximum(np.ceil(xy.min(axis=1)).astype(np.int32), [0, 0])
    high = np.minimum(np.floor(xy.max(axis=1)).astype(np.int32), [width - 1, height - 1])
    sizes = np.maximum(0, high - low + 1)
    counts = sizes[:, 0] * sizes[:, 1]
    active = np.flatnonzero(counts)
    buffer = np.full(width * height, np.inf, np.float32)
    owner = np.full(width * height, -1, np.int32)
    for start in range(0, len(active), 8192):
        group = active[start:start + 8192]
        total = np.cumsum(counts[group])
        cuts = np.unique(np.r_[0, np.searchsorted(total, np.arange(420000, total[-1], 420000)), len(group)])
        for a, b in zip(cuts, cuts[1:]):
            faces = group[a:b]
            local_counts = counts[faces]
            lookup = np.repeat(faces, local_counts)
            offset = np.arange(len(lookup)) - np.repeat(np.cumsum(local_counts) - local_counts, local_counts)
            px = low[lookup, 0] + offset % sizes[lookup, 0]
            py = low[lookup, 1] + offset // sizes[lookup, 0]
            corners = xy[faces]
            e1 = corners[:, 1] - corners[:, 0]
            e2 = corners[:, 2] - corners[:, 0]
            determinant = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            inv = np.divide(1., determinant, where=np.abs(determinant) > 1e-10,
                            out=np.zeros_like(determinant))
            inv1 = np.repeat(e2 * inv[:, None], local_counts, axis=0)
            inv2 = np.repeat(e1 * inv[:, None], local_counts, axis=0)
            dx = px - xy[lookup, 0, 0]
            dy = py - xy[lookup, 0, 1]
            v = dx * inv1[:, 1] - dy * inv1[:, 0]
            w = dy * inv2[:, 0] - dx * inv2[:, 1]
            ok = (v >= -.00002) & (w >= -.00002) & (v + w <= 1.00002) & np.repeat(inv != 0., local_counts)
            ids = lookup[ok]
            pixels = (py * width + px)[ok]
            z = depths[ids, 0] + v[ok] * (depths[ids, 1] - depths[ids, 0]) + w[ok] * (depths[ids, 2] - depths[ids, 0])
            z = z.astype(np.float32)
            np.minimum.at(buffer, pixels, z)
            wins = z == buffer[pixels]
            owner[pixels[wins]] = ids[wins]
    return buffer.reshape(shape), owner.reshape(shape)


def axis_extents(low, high, origin, axis):
    center = (low + high) / 2 - origin
    radius = (high - low) / 2 @ np.abs(axis)
    value = center @ axis
    return value - radius, value + radius


def make_geometry(contract, scene, out, ppm):
    path = np.array(contract['route']['enu'])[:, :2]
    origin = path[0]
    delta = path[-1] - origin
    length = np.linalg.norm(delta)
    along = delta / length
    across = np.array([-along[1], along[0]])
    eye_z = np.array([v['eyeEnu'][2] for v in contract['train']])
    zmin, zmax = np.floor(eye_z.min() - 5), np.ceil(eye_z.max() + 22)
    triangles, sides, owners, names = [], [], [], []
    with GltfStream(scene, cache_size=0) as reader:
        for row in reader.rows:
            name = row['node']
            if name.split('__')[0] not in ('buildings', 'facades', 'foundations'):
                continue
            low, high = row['minSource'], row['maxSource']
            t0, t1 = axis_extents(low[:2], high[:2], origin, along)
            d0, d1 = axis_extents(low[:2], high[:2], origin, across)
            if t1 < -2 or t0 > length + 2 or d1 < -36 or d0 > 36 or high[2] < zmin or low[2] > zmax:
                continue
            prim = row['primitive']
            pos = np.array(reader._accessor(prim['attributes']['POSITION']), copy=True)
            matrix = GLB_TO_SOURCE @ row['transform']
            vrt = pos @ matrix[:3, :3].T + matrix[:3, 3]
            idx = reader._accessor(prim['indices']).flatten() if 'indices' in prim else np.arange(len(pos))
            tri = vrt[idx.reshape(-1, 3)]
            center = tri.mean(axis=1)
            offset = center[:, :2] - origin
            d = offset @ across
            t = offset @ along
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            lens = np.linalg.norm(normals, axis=1)
            normal = normals / np.maximum(lens[:, None], 1e-9)
            side = np.sign(d)
            ok = ((t > -2) & (t < length + 2) & (np.abs(d) > 3.5) & (np.abs(d) < 36) &
                  (center[:, 2] > zmin - 1.) & (center[:, 2] < zmax + 2.) &
                  (lens > .000015) & ((normal[:, :2] @ across) * -side > .28))
            if not ok.any():
                continue
            triangles.append(tri[ok].astype(np.float32))
            sides.append(side[ok].astype(np.int8))
            owners.append(np.full(ok.sum(), len(names), np.int32))
            names.append(name)
    if not triangles:
        raise ValueError('The selected route has no eligible street-facing facade geometry.')
    tri = np.concatenate(triangles)
    side = np.concatenate(sides)
    np.savez_compressed(out / 'block-surfaces.npz', triangles=tri, side=side, along=along,
                        origin=origin, length=length, node=np.concatenate(owners), nodeNames=names)
    width = int(np.ceil(length * ppm)) + 4
    height = int(np.ceil((zmax - zmin) * ppm)) + 4
    smin, ztop = -2 / ppm, zmax + 2 / ppm
    for sign in (-1, 1):
        ids = np.flatnonzero(side == sign)
        corners = tri[ids]
        delta_xy = corners[:, :, :2] - origin
        uv = np.stack([((delta_xy @ along) - smin) * ppm - .5,
                       (ztop - corners[:, :, 2]) * ppm - .5], axis=-1)
        depth, owner = raster(uv, sign * (delta_xy @ across), (height, width))
        yy, xx = np.indices((height, width), dtype=np.float32)
        valid = np.isfinite(depth) & (xx >= 2) & (xx < width - 2) & (yy >= 2) & (yy < height - 2)
        depth[~valid] = 0.
        ids_view = np.where(valid, ids[owner], -1).astype(np.int32)
        s = smin + (xx + .5) / ppm
        xy = origin + s[..., None] * along + (sign * depth)[..., None] * across
        xyz = np.dstack([xy, ztop - (yy + .5) / ppm]).astype(np.float32)
        xyz[~valid] = np.nan
        np.savez_compressed(out / f'surface-side-{sign}.npz', xyz=xyz, validMask=valid, side=sign,
                            face=ids_view, ppm=ppm, smin=smin, ztop=ztop, shape=[height, width])
        print(f'Side {sign} chart {width} x {height}, {valid.sum()} surface texels', flush=True)
    with scene.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    evidence = dict(scenePath=str(scene.resolve()), sceneSha256=digest, candidateTriangles=len(tri), sceneNodes=len(names),
                    axes='All support arrays are east, north, up in scene metres. The GLB overlay is east, up, minus north.',
                    chart='Side is sign of offset along (-alongY,alongX). Rows descend from ztop. Column s = smin+(x+.5)/ppm.',
                    spacingM=1/ppm, corridorWidthEachSideM=36., routeLengthM=length,
                    algorithm='Extract original outward-facing R31 building, facade and foundation triangles. Resolve nearest signed '
                              'depth and ownership from street towards each facade, at each metric orthographic chart sample.')
    (out / 'photo-geometry-receipt.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'{len(tri)} front-facing triangles, {len(names)} scene nodes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cameras', type=Path)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ppm', type=float, default=24.)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    make_geometry(json.loads(args.cameras.read_text()), args.scene, args.output, args.ppm)
