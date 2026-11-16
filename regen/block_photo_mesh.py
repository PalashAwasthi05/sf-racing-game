"""Bake independent-view photographic atlases onto original block mesh triangles.

The fit consumes cameras.train exclusively. For each chart sample, reject
occluded and non-building source pixels, score view incidence and distance,
select a coherent source per facade, and blend at source seams. Held-out
photographs are read only after the run has exported and hashed the GLB.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from frontage_crops_v15 import world_dirs_to_pano_uv
from regen.block_photo_mesh_geometry import raster

ROOT = Path(__file__).resolve().parents[1]
VIEW_W, VIEW_H = 960, 640
HFOV = 124.
PITCH = 13.


class FacadeMask:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        torch.set_num_threads(2)
        cached = ROOT / '.models/models--nvidia--segformer-b0-finetuned-cityscapes-1024-1024/snapshots'
        snapshot = next(cached.iterdir())
        self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            snapshot, use_safetensors=False, local_files_only=True).eval()
        self.keep = [key for key, name in self.model.config.id2label.items() if name in ('building', 'wall')]

    def __call__(self, rgb):
        import torch
        small = cv2.resize(rgb, (768, 512), interpolation=cv2.INTER_AREA)
        values = self.processor(images=small, do_resize=False, return_tensors='pt').pixel_values
        with torch.inference_mode():
            logits = self.model(pixel_values=values).logits
            probs = torch.softmax(logits, dim=1)[0, self.keep].sum(dim=0).numpy()
        return cv2.resize(probs, (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)


def calc_basis(across, side):
    horizontal = np.r_[across * side, 0.]
    f = np.cos(np.radians(PITCH)) * horizontal + np.array([0., 0., np.sin(np.radians(PITCH))])
    right = np.cross(f, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.float32([right, down, f])


FOCAL = VIEW_W / (2 * np.tan(np.radians(HFOV) / 2))
CY, CX = (VIEW_H - 1) / 2, (VIEW_W - 1) / 2


def sample_pano(pano, dirs, rotation):
    u, v = world_dirs_to_pano_uv(dirs, rotation)
    return cv2.remap(pano, (u * pano.shape[1] - .5).astype('f4'), (v * pano.shape[0] - .5).astype('f4'),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def perspective(pano, rotation, basis):
    y, x = np.mgrid[:VIEW_H, :VIEW_W].astype('f4')
    rays = np.stack([(x - CX) / FOCAL, (y - CY) / FOCAL, np.ones_like(x)], axis=-1) @ basis
    return sample_pano(pano, rays, rotation)


def screen(vectors, basis):
    view = vectors @ basis.T
    z = view[..., 2]
    uv = view[..., :2] * (FOCAL / np.maximum(z, .01))[..., None] + [CX, CY]
    return uv.astype('f4'), z


def projection_depth(tri, eye, basis):
    uv, z = screen(tri - eye, basis)
    valid = (z > .45).all(axis=1)
    inv, _ = raster(uv[valid], -1 / z[valid], (VIEW_H, VIEW_W))
    return np.divide(-1., inv, where=np.isfinite(inv), out=np.full_like(inv, 1e4))


def reproject_view(record, pano, side, geom, chart, mask_model):
    along = geom['along']
    across = np.array([-along[1], along[0]])
    basis = calc_basis(across, side)
    eye = np.array(record['eyeEnu'], dtype='f4')
    rot = np.array(record['rotationWorldToCamera'], dtype='f4')
    view = perspective(pano, rot, basis)
    semantic = mask_model(view)
    tri = geom['triangles'][geom['side'] == side]
    zbuf = projection_depth(tri, eye, basis)
    anchor = (eye[:2] - geom['origin']) @ along
    h, w = chart['validMask'].shape
    left = max(0, int((anchor - 44 - chart['smin']) * chart['ppm']))
    right = min(w, int(np.ceil((anchor + 44 - chart['smin']) * chart['ppm'])))
    span = slice(left, right)
    valid = chart['validMask'][:, span]
    xyz = chart['xyz'][:, span]
    rel = np.where(valid[..., None], xyz - eye, 0.)
    uv, z = screen(rel, basis)
    # Use target geometry and the independent camera z-buffer to reject
    # receding/overlapping surfaces hidden in this particular source view.
    uv = np.clip(uv, -16000, 16000)
    depth = cv2.remap(zbuf, uv, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    prob = cv2.remap(semantic, uv, None, cv2.INTER_LINEAR)
    rng = np.linalg.norm(rel, axis=2)
    normals = geom['normal'][np.maximum(chart['face'][:, span], 0)]
    incidence = -(rel * normals).sum(axis=2) / np.maximum(rng, .001)
    ok = (valid & (prob > .68) & (z > .5) & (z <= depth + .20) & (incidence > .20) & (rng <= 48.) &
          (uv[..., 0] > 2) & (uv[..., 0] < VIEW_W - 3) & (uv[..., 1] > 2) & (uv[..., 1] < VIEW_H - 3))
    rgb = sample_pano(pano, rel, rot)
    quality = np.maximum(0., incidence) ** 1.25 * (16. / np.maximum(rng, 7.)) ** 1.25 * prob
    weight = np.where(ok, np.clip(quality, .015, 3), 0).astype('f4')
    return dict(span=span, weight=weight, rgb=rgb, index=record['id'], valid=valid, xyz=xyz, uv=uv,
                view=view, semantic=semantic, depth=zbuf, referenceDepth=z, basis=basis, eye=eye)


def fuse(frames, geom, ch):
    shape = ch['validMask'].shape
    names = [str(s).split('__', 2)[1] for s in geom['nodeNames']]
    features, node_to_feature = np.unique(names, return_inverse=True)
    feature = node_to_feature[geom['node'][np.maximum(ch['face'], 0)]]
    feature_count = len(features)
    source_scores = []
    support = np.zeros(shape, np.uint8)
    primary = np.zeros(shape, np.int16)
    top = np.zeros(shape, np.float32)
    for frame in frames:
        s = frame['span']
        weights = frame['weight']
        ids = feature[:, s]
        supported = weights > 0
        support[:, s] += supported
        count = np.bincount(ids[supported], weights=weights[supported], minlength=feature_count)
        source_scores.append(count)
    scores = np.array(source_scores)
    rank = scores / np.maximum(1e-6, scores.max(axis=0))
    for i, frame in enumerate(frames):
        s = frame['span']
        # A continuous source choice across one architectural object reduces
        # parallax doubles; valid alternatives fill its occlusions.
        q = frame['weight'] * (.1 + .9 * rank[i, feature[:, s]] ** 4)
        better = q > top[:, s]
        top[:, s][better] = q[better]
        primary[:, s][better] = i
    color = np.zeros((*shape, 3), np.float32)
    mass = np.zeros(shape, np.float32)
    low_sum = np.zeros_like(color)
    low_weight = np.zeros_like(mass)
    main_low = np.zeros_like(color)
    contributions = []
    for i, frame in enumerate(frames):
        s = frame['span']
        weight = frame['weight']
        valid = weight > 0
        primary_here = (primary[:, s] == i) & valid
        seam = cv2.GaussianBlur(primary_here.astype('f4'), (0, 0), 2.1) * valid
        seam += .08 * primary_here
        raw = frame['rgb'].astype('f4')
        # Only low spatial frequency illumination is averaged away from a seam.
        smooth = cv2.GaussianBlur(raw, (0, 0), 3.2)
        low_sum[:, s] += smooth * weight[..., None]
        low_weight[:, s] += weight
        color[:, s] += raw * seam[..., None]
        main_low[:, s] += smooth * seam[..., None]
        mass[:, s] += seam
        contributions.append(dict(id=frame['index'], candidatePixels=int(valid.sum()),
                                  primaryPixels=int(primary_here.sum())))
    main = color / np.maximum(mass[..., None], 1e-6)
    delta = low_sum / np.maximum(low_weight[..., None], 1e-6) - main_low / np.maximum(mass[..., None], 1e-6)
    rgb = np.uint8(np.clip(np.rint(main + .24 * np.clip(delta, -42., 42.)), 0, 255))
    # A narrow color bleed only outside the opaque support avoids dark mip edges.
    from scipy.ndimage import distance_transform_edt
    valid = support > 0
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    bleed = (~valid) & (distance <= 8)
    rgb[bleed] = rgb[nearest[0][bleed], nearest[1][bleed]]
    rgba = np.dstack([rgb, np.uint8(valid) * 255])
    return rgba, support, contributions


def write_glb(path, materials):
    """One unlit glTF primitive per side, indexed original vertices, PNG alpha."""
    blob = bytearray()
    views, accessors, images, textures, mats, meshes, nodes = [], [], [], [], [], [], []

    def add_view(data, target=None):
        blob.extend(b'\0' * ((-len(blob)) % 4))
        entry = dict(buffer=0, byteOffset=len(blob), byteLength=len(data))
        if target:
            entry['target'] = target
        views.append(entry)
        blob.extend(data)
        return len(views) - 1

    def accessor(data, typ, component, target, bounds=False):
        view = add_view(data.tobytes(), target)
        ent = dict(bufferView=view, componentType=component, count=len(data), type=typ)
        if bounds:
            ent.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
        accessors.append(ent)
        return len(accessors) - 1

    for side, tri, texcoord, rgba in materials:
        pixels = io.BytesIO()
        Image.fromarray(rgba).save(pixels, format='PNG', compress_level=4)
        img = add_view(pixels.getvalue())
        images.append(dict(bufferView=img, mimeType='image/png'))
        i = len(textures)
        textures.append(dict(source=i, sampler=0))
        mats.append(dict(name=f'Photographic facade {side}', alphaMode='MASK', alphaCutoff=.5,
                         pbrMetallicRoughness=dict(baseColorTexture=dict(index=i), metallicFactor=0, roughnessFactor=1),
                         extensions={'KHR_materials_unlit': {}}))
        # No vertex offset. The viewer uses polygon offset in its overlay pass.
        glb_xyz = np.stack([tri[..., 0], tri[..., 2], -tri[..., 1]], axis=-1)
        packed = np.hstack([glb_xyz.reshape(-1, 3), texcoord.reshape(-1, 2)]).astype('<f4')
        unique, indices = np.unique(packed, axis=0, return_inverse=True)
        p = accessor(np.ascontiguousarray(unique[:, :3]), 'VEC3', 5126, 34962, True)
        uv = accessor(np.ascontiguousarray(unique[:, 3:]), 'VEC2', 5126, 34962)
        ind = accessor(indices.astype('<u4').ravel(), 'SCALAR', 5125, 34963)
        meshes.append(dict(name=f'R31 wall relief side {side}', primitives=[dict(attributes=dict(POSITION=p, TEXCOORD_0=uv),
                                                                                  indices=ind, material=i)]))
        nodes.append(dict(mesh=i, name=f'Photo wall side {side}'))
    doc = dict(asset=dict(version='2.0', generator='R31 block photo mesh'), extensionsUsed=['KHR_materials_unlit'],
               scene=0, scenes=[dict(nodes=list(range(len(nodes))))], nodes=nodes, meshes=meshes, materials=mats,
               images=images, textures=textures, samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
               bufferViews=views, accessors=accessors, buffers=[dict(byteLength=len(blob))])
    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' ' * ((-len(js)) % 4)
    blob.extend(b'\0' * ((-len(blob)) % 4))
    data = (struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(blob)) +
            struct.pack('<I4s', len(js), b'JSON') + js + struct.pack('<I4s', len(blob), b'BIN\0') + blob)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def input_geometry(out):
    geometry = dict(np.load(out / 'block-surfaces.npz'))
    tri = geometry['triangles']
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    geometry['normal'] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    charts = {side: dict(np.load(out / f'surface-side-{side}.npz')) for side in (-1, 1)}
    return geometry, charts


def coverage(arr, mask):
    return round(float(arr[mask].mean()), 5) if mask.any() else None


def fit(contract, geom, charts, out, model):
    train = contract['train']
    mats, metrics = [], []
    for side, ch in charts.items():
        frames = []
        for rec in train:
            pano = cv2.cvtColor(cv2.imread(rec['imagePath']), cv2.COLOR_BGR2RGB)
            projected = reproject_view(rec, pano, side, geom, ch, model)
            frames.append({k: projected[k] for k in ('span', 'weight', 'rgb', 'index')})
            del projected, pano
        rgba, support, contributions = fuse(frames, geom, ch)
        del frames
        h, w = support.shape
        ids = np.unique(ch['face'][(support > 0) & ch['validMask']])
        tri = geom['triangles'][ids]
        xy = (tri[..., :2] - geom['origin']) @ geom['along']
        uv = np.stack([((xy - ch['smin']) * ch['ppm']) / w,
                       ((ch['ztop'] - tri[..., 2]) * ch['ppm']) / h], axis=-1)
        mats.append((side, tri, uv, rgba))
        Image.fromarray(rgba).save(out / f'photo-atlas-side-{side}.png', compress_level=4)
        frac = coverage(support > 0, ch['validMask'])
        metrics.append(dict(side=int(side), originalTrianglesWithPhoto=int(len(ids)), visibleFacadeTexels=int(ch['validMask'].sum()),
                            texturedTexels=int((support > 0).sum()), coverageFraction=frac,
                            multiViewFractionOfTextured=coverage(support >= 2, support > 0), atlasSize=[w, h],
                            texelsPerMeter=float(ch['ppm']), sourceContributions=contributions))
        print(f'Baked side {side} coverage {frac}, {len(ids)} original triangles', flush=True)
    return write_glb(out / 'photo-mesh.glb', mats), metrics


def choose_training_pose(train, option, geometry, charts, model):
    """Compare unaltered and refined poses on common training-only wall samples."""
    proposed = json.loads(option.read_text())
    if not proposed.get('used') or set(proposed['cameras']) != {r['id'] for r in train}:
        return train, dict(accepted=False, reason='Missing complete accepted pose proposal')
    refined = [{**r, **proposed['cameras'][r['id']]} for r in train]
    across = np.array([-geometry['along'][1], geometry['along'][0]])
    pair_stats = []
    for side in (-1, 1):
        basis = calc_basis(across, side)
        ch = charts[side]
        xyz = ch['xyz'][::10, ::10]
        known = ch['validMask'][::10, ::10]
        prev = None
        for raw, alt in zip(train, refined):
            pano = cv2.cvtColor(cv2.imread(raw['imagePath'], cv2.IMREAD_REDUCED_COLOR_2), cv2.COLOR_BGR2RGB)
            probes = []
            for cfg in (raw, alt):
                eye = np.float32(cfg['eyeEnu'])
                rot = np.float32(cfg['rotationWorldToCamera'])
                rel = np.where(known[..., None], xyz-eye, 0.)
                uv, z = screen(rel, basis)
                source = sample_pano(pano, rel, rot)
                view = perspective(pano, rot, basis)
                prob = cv2.remap(model(view), np.clip(uv,-16000,16000), None, cv2.INTER_LINEAR)
                ok = known & (prob > .8) & (z > 3.) & (np.linalg.norm(rel,axis=2) < 30.)
                probes.append((source, ok))
            if prev is not None:
                ok = probes[0][1] & probes[1][1] & prev[0][1] & prev[1][1]
                if ok.sum() >= 50:
                    err = []
                    for k in (0, 1):
                        a, b = probes[k][0][ok].astype('f4'), prev[k][0][ok].astype('f4')
                        diff = a-b
                        corrected = diff - np.median(diff,axis=0)
                        err.append(float(np.median(np.abs(corrected).mean(axis=1))))
                    pair_stats.append(dict(side=side, id=raw['id'], samples=int(ok.sum()), originalMAE=err[0], refinedMAE=err[1]))
            prev = probes
    original = float(np.median([v['originalMAE'] for v in pair_stats]))
    candidate = float(np.median([v['refinedMAE'] for v in pair_stats]))
    accepted = candidate <= original
    audit = dict(accepted=accepted, poseProposal=option.name, pairs=len(pair_stats), originalMedian=original,
                 refinedMedian=candidate, perPair=pair_stats, sample='Each tenth chart pixel; common high-confidence photographic building '
                 'pixels at <30 m in both adjacent train frames and both pose hypotheses. Remove per-pair channel medians and compare '
                 'median absolute sRGB error per pair. No held-out drive used.')
    return refined if accepted else train, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cameras', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--train-poses', type=Path)
    args = parser.parse_args()
    out = args.output
    contract = json.loads(args.cameras.read_text())
    train = contract['train']
    assert not ({r['id'] for r in train} & {r['id'] for r in contract['heldout']})
    assert not ({r['driveId'] for r in train} & {r['driveId'] for r in contract['heldout']})
    geom, charts = input_geometry(out)
    model = FacadeMask()
    pose_check = None
    if args.train_poses:
        train, pose_check = choose_training_pose(train, args.train_poses, geom, charts, model)
        contract = {**contract, 'train': train}
        (out / 'photo-train-pose-check.json').write_text(json.dumps(pose_check, indent=2) + '\n')
        print(f'Pose refinement accepted {pose_check["accepted"]}', flush=True)
    digest, metrics = fit(contract, geom, charts, out, model)
    total = sum(row['visibleFacadeTexels'] for row in metrics)
    covered = sum(row['texturedTexels'] for row in metrics)
    report = dict(schema='metric-facade-photoprojection-v1', mesh='photo-mesh.glb', sha256=digest,
                  sceneBasis='Final R31-street original triangles. Source ENU to glTF [east,up,-north].',
                  fitDriveIds=sorted({r['driveId'] for r in train}), trainFrames=len(train),
                  trainPoseRefinement=pose_check and {k:v for k,v in pose_check.items() if k != 'perPair'},
                  texture='Camera ray/equirectangular reprojection to actual R31 triangles, independent camera depth test, '
                          'Cityscapes building/wall mask, incidence/distance and facade continuity source selection, '
                          'multi-view low-frequency color and narrow seam fusion. glTF unlit PNG alpha-mask atlases.',
                  coverageDenominator='Street-facing R31 envelope samples rendered outward on each side at fixed metric density. '
                                      'Excludes gaps and side-facing normals. Includes occluded targets. Geometry coverage differs '
                                      'from driving-camera pixel coverage.',
                  candidateTriangles=int(len(geom['triangles'])), facadeCoverageFraction=covered / total,
                  geometryProvenance='photo-geometry-receipt.json', independentDriveEvaluation='photo-mesh-heldout-metrics.json',
                  sides=metrics, limits=['Unsurveyed panorama registration and approximate R31 mesh relief.',
                                         'Captured illumination and reflections are baked into wall appearance.',
                                         'Occluded surface samples fall through to the normal R31 material.'])
    (out / 'photo-mesh-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Exported {out / "photo-mesh.glb"} {digest}', flush=True)


if __name__ == '__main__':
    main()
