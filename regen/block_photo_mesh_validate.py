"""Render the emitted GLB against the held-out drive from its recorded cameras.

This executable only reads the fitted GLB and scene support. Its comparison
reports camera-visible surface coverage and error on the same photographic
building pixels. Neither those pixels nor the results are returned to the fit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from regen.block_photo_mesh import (FacadeMask, VIEW_H, VIEW_W, calc_basis, input_geometry,
                                   perspective, projection_depth, screen)
from regen.block_photo_mesh_geometry import raster


def read_overlay(path):
    data = path.read_bytes()
    _, version, n = struct.unpack_from('<4sII', data)
    assert version == 2 and len(data) == n
    jlen = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + jlen])
    assert doc['extensionsUsed'] == ['KHR_materials_unlit']
    offset = 28 + jlen

    def block(ix):
        b = doc['bufferViews'][ix]
        start = offset + b.get('byteOffset', 0)
        return data[start:start + b['byteLength']]

    def accessor(ix):
        a = doc['accessors'][ix]
        dtype = {5125:'<u4', 5126:'<f4'}[a['componentType']]
        return np.frombuffer(block(a['bufferView']), dtype=dtype).reshape(a['count'], -1)

    overlays = {}
    for mesh in doc['meshes']:
        side = int(mesh['name'].split()[-1])
        prim = mesh['primitives'][0]
        ids = accessor(prim['indices']).reshape(-1, 3)
        pos = accessor(prim['attributes']['POSITION'])[ids]
        xyz = np.stack([pos[...,0], -pos[...,2], pos[...,1]], axis=-1)
        uv = accessor(prim['attributes']['TEXCOORD_0'])[ids]
        mat = doc['materials'][prim['material']]
        tex = doc['textures'][mat['pbrMetallicRoughness']['baseColorTexture']['index']]
        image = doc['images'][tex['source']]
        rgba = np.asarray(Image.open(io.BytesIO(block(image['bufferView']))).convert('RGBA'))
        assert np.isfinite(xyz).all() and np.isfinite(uv).all()
        overlays[side] = (xyz, uv, rgba)
    return overlays, hashlib.sha256(data).hexdigest()


def photo_render(overlay, eye, basis):
    tri, uv, tex = overlay
    px, z = screen(tri - eye, basis)
    ok = (z > .45).all(axis=1)
    ids = np.flatnonzero(ok)
    inverse, face = raster(px[ok], -1 / z[ok], (VIEW_H, VIEW_W))
    yy, xx = np.nonzero(face >= 0)
    tid = ids[face[yy, xx]]
    corners = px[tid]
    delta1 = corners[:, 1] - corners[:, 0]
    delta2 = corners[:, 2] - corners[:, 0]
    dx, dy = xx-corners[:, 0, 0], yy-corners[:, 0, 1]
    determinant = delta1[:, 0] * delta2[:, 1] - delta1[:, 1] * delta2[:, 0]
    b = (dx * delta2[:, 1] - dy * delta2[:, 0]) / determinant
    c = (dy * delta1[:, 0] - dx * delta1[:, 1]) / determinant
    weights = np.column_stack([1 - b - c, b, c]) / z[tid]
    weights /= weights.sum(axis=1)[:, None]
    coords = np.sum(uv[tid] * weights[..., None], axis=1)
    pxmap = np.full((VIEW_H,VIEW_W,2), -1., dtype='f4')
    pxmap[yy,xx] = coords * [tex.shape[1], tex.shape[0]] - .5
    color = cv2.remap(tex, pxmap, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    color[face < 0, 3] = 0
    depth = np.divide(-1., inverse, where=np.isfinite(inverse), out=np.full_like(inverse, 1e4))
    return color, depth


def validate(contract, geom, out):
    overlays, sha = read_overlay(out / 'photo-mesh.glb')
    report = json.loads((out / 'photo-mesh-report.json').read_text())
    assert report['sha256'] == sha
    assert {r['driveId'] for r in contract['heldout']}.isdisjoint(report['fitDriveIds'])
    selector = FacadeMask()
    records, examples, error_sum, squared_sum, total = [], [], 0., 0., 0
    possible_total = 0
    show = {round(f*(len(contract['heldout'])-1)) for f in (.18,.5,.82)}
    across = np.array([-geom['along'][1], geom['along'][0]])
    neutral = np.array([52,57,62], np.uint8)
    for i, r in enumerate(contract['heldout']):
        pano = cv2.cvtColor(cv2.imread(r['imagePath']), cv2.COLOR_BGR2RGB)
        eye = np.float32(r['eyeEnu'])
        for side in (-1,1):
            basis = calc_basis(across, side)
            obs = perspective(pano, np.float32(r['rotationWorldToCamera']), basis)
            sem = selector(obs) > .68
            pred, depth = photo_render(overlays[side], eye, basis)
            base_depth = projection_depth(geom['triangles'][geom['side'] == side], eye, basis)
            possible = sem & (base_depth < 100.)
            eligible = possible & (pred[...,3] >= 128) & (depth <= base_depth + .20)
            err = pred[eligible, :3].astype('f4') - obs[eligible].astype('f4')
            count = int(eligible.sum())
            possible_total += int(possible.sum())
            error_sum += np.abs(err).sum(dtype=np.float64)
            squared_sum += np.square(err).sum(dtype=np.float64)
            total += count
            records.append(dict(imageId=r['id'], side=side, scoredPixels=count, observedBuildingOnGeometryPixels=int(possible.sum()),
                                coverage=round(count/max(1,possible.sum()),4), meanAbsoluteError8Bit=round(float(np.abs(err).mean()),2) if count else None))
            if i in show:
                rendered = np.where(((pred[...,3] >= 128) & (depth <= base_depth+.20))[..., None], pred[..., :3], neutral)
                residual = np.mean(np.abs(pred[...,:3].astype('f4')-obs), axis=2)
                heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.round(np.clip(residual/100,0,1)*255)),cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                heat[~eligible] = neutral
                parts = [Image.fromarray(im).resize((384,256),Image.Resampling.LANCZOS) for im in (obs,rendered,heat)]
                examples.append((r['id'], side, records[-1], parts))
    sheet = Image.new('RGB', (1168, 62 + len(examples) * 284), (245,246,244))
    draw = ImageDraw.Draw(sheet)
    draw.text((12,12),'H E L D - O U T   D R I V E    /    P H O T O G R A P H I C   M E S H',fill=(28,35,36))
    for x,title in zip((8,396,784),('Independent photograph','Emitted overlay on its actual triangles','Mean RGB error   /   dark = low, yellow = 100+')):
        draw.text((x,39),title,fill=(35,42,44))
    for i,(rid, side, rec, parts) in enumerate(examples):
        y = 60+i*284
        for x,im in zip((8,396,784),parts):
            sheet.paste(im,(x,y))
        draw.text((8,y+260),f'Held-out {rid}    /    side {side:+}    /    {100*rec["coverage"]:.0f}% photographed coverage    /    RGB MAE {rec["meanAbsoluteError8Bit"]}',fill=(35,42,44))
    sheet.save(out / 'photo-mesh-heldout-comparison.jpg', quality=89)
    result = dict(schema='photo-mesh-independent-drive-replay-v1', scoredGlbSha256=sha,
                  source='Emitted GLB readback, perspective-correct UV interpolation. Original geometry depth test.',
                  evaluationDriveIds=sorted({r['driveId'] for r in contract['heldout']}), heldoutFrames=len(contract['heldout']), views=len(records),
                  imageSize=[VIEW_W, VIEW_H], fieldOfViewHorizontalDegrees=124, pitchDegrees=13,
                  scoredPixels=total, observedBuildingOnGeometryPixels=possible_total, scoredCoverage=total/possible_total,
                  meanAbsoluteError8Bit=error_sum / (total*3), psnrDb=10*math.log10(255**2 / (squared_sum/(total*3))),
                  interpretation='Only registered held-out photograph pixels classified building/wall and lying on the selected facade geometry '
                                 'enter the denominator. The color score uses overlapping opaque source texture only. Raw sRGB differences '
                                 'also include lighting, scene changes and independent camera alignment across the two years.',
                  samples='Fixed 18%, 50% and 82% positions of the held-out traversal, both sides.', perView=records)
    (out / 'photo-mesh-heldout-metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'perView'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cameras', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    geom, _ = input_geometry(args.output)
    validate(json.loads(args.cameras.read_text()), geom, args.output)
