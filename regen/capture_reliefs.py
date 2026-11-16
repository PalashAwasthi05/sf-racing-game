"""Recover visible relief outlines from registered photographs, not a motif library.

Input is any compiled profile set with source provenance. Every relief goes through
the same bounded detector and contrast/occlusion checks. Output is a profile copy
plus unaccepted proposals. Contrast masks can confuse lighting with sculpture,
so they must not enter production geometry without independent silhouette
validation. No geometry is selected by a building identity.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

VERSION = 'relief-outline-4-proposals-only'


def detection_key(image_bytes, state, questions):
    """A different source region is a different inference, even on the same photo."""
    payload = json.dumps({'state': state, 'questions': questions}, sort_keys=True).encode()
    return hashlib.sha256(VERSION.encode() + image_bytes + payload).hexdigest()


def split_light_relief(rgb, isolated=True):
    """Conservative light-relief mask. Reject low contrast/solid plaques.

    This is deliberately restricted to light ornament on darker backing. The
    semantic detector must establish that premise independently. Separate shadow,
    backing and highlights: a two-class split often mistakes sunlit backing for
    cream sculpture because deep cast shadows dominate its dark class.
    """
    gray = np.asarray(rgb, float) @ np.array([.2126, .7152, .0722])
    if min(gray.shape) < 12:
        return None, {'reason': 'insufficient-source-pixels'}
    centers = np.percentile(gray, [10, 50, 95])
    for _ in range(32):
        assignment = np.argmin(abs(gray[..., None] - centers), axis=-1)
        updated = np.array([np.mean(gray[assignment == k]) if np.any(assignment == k) else centers[k] for k in range(3)])
        if np.max(abs(updated - centers)) < .05: break
        centers = np.sort(updated)
    threshold = (centers[1] + centers[2]) / 2
    mask = gray > threshold
    if not mask.any() or mask.all(): return None, {'reason': 'no-separable-foreground'}
    contrast = float(np.median(gray[mask]) - np.median(gray[~mask]))
    if contrast < 28: return None, {'reason': 'weak-foreground-contrast', 'contrast': contrast}
    labels, count = ndimage.label(mask)
    areas = np.bincount(labels.ravel())
    keep = areas >= max(3, mask.size * .001)
    keep[0] = False
    if isolated:
        # An isolated sculpture sits inside its context. Neighboring cornices
        # and window frames typically run out of that region; exclude those
        # connected pixels rather than turning the framing into sculpture.
        border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        keep[border] = False
    mask = keep[labels]
    yy, xx = np.nonzero(mask)
    if len(xx) < 25 or np.ptp(xx) < 10 or np.ptp(yy) < 10:
        return None, {'reason': 'insufficient-isolated-relief-support'}
    # Context size is arbitrary; occupancy must use the recovered object bounds.
    fill = float(mask.sum() / ((np.ptp(xx) + 1) * (np.ptp(yy) + 1)))
    if not .06 < fill < .70:
        return None, {'reason': 'foreground-occupancy-outside-relief-range', 'fill': fill}
    # Long filled border runs usually mean a cornice/window frame got included.
    edge_fill = max(float(mask[:2].mean()), float(mask[-2:].mean()),
                    float(mask[:, :2].mean()), float(mask[:, -2:].mean()))
    if edge_fill > .75:
        return None, {'reason': 'foreground-truncated-or-frame-included', 'edgeFill': edge_fill}
    return mask, {'threshold': float(threshold), 'contrast': contrast, 'fill': fill,
                  'edgeFill': edge_fill, 'components': int(count)}


def vector_mask(mask):
    """Unit-square polygons, y upward; simplify below half a source pixel."""
    h, w = mask.shape
    parts = [shape(geom) for geom, value in shapes(mask.astype('uint8'), mask=mask) if value == 1]
    from shapely.affinity import affine_transform
    region = unary_union(parts).simplify(.28, preserve_topology=True)
    region = affine_transform(region, [1 / w, 0, 0, -1 / h, 0, 1])
    return mapping(region)


def capture(profile, relief, area, registered, cache):
    prov = profile.get('readingProvenance') or {}
    if not prov.get('registeredCropSource') or prov.get('status') != 'ok':
        return None, {'status': 'unresolved', 'reason': 'registered-source-required'}
    side = registered.get(profile['id'].removeprefix('regen-'))
    if not side or side.get('status') != 'ok':
        return None, {'status': 'unresolved', 'reason': 'ambiguous-side-source'}
    piece = side['piece']
    views = [v for v in piece['views'] if v.get('rect') == prov.get('rect') and v.get('image') == prov.get('image')]
    if len(views) != 1:
        return None, {'status': 'unresolved', 'reason': 'ambiguous-view-source'}
    view = views[0]; pad, ppm = view['rectPad'], view['ppm']
    source = side['path'].parent / 'img' / view['rect']
    image = Image.open(source).convert('RGB')
    W, H = piece['lengthM'], piece['heightM']
    x0, z0, x1, z1 = relief['rect']
    # The model is shown a bounded neighborhood; it cannot choose an unrelated
    # feature elsewhere on the building. Projected bays remain uncertain in depth.
    expected = [pad + x0 * W * ppm, pad + (H - z1 * H) * ppm,
                pad + x1 * W * ppm, pad + (H - z0 * H) * ppm]
    dx = max(12, (expected[2] - expected[0]) * .2)
    dy = max(12, (expected[3] - expected[1]) * .3)
    bounds = [max(pad, int(expected[0] - dx)), max(pad, int(expected[1] - dy)),
              min(image.width - pad, int(expected[2] + dx)), min(image.height - pad, int(expected[3] + dy))]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None, {'status': 'unresolved', 'reason': 'source-region-outside-image'}
    crop = image
    region = {'left': bounds[0] / image.width, 'top': bounds[1] / image.height,
              'right': bounds[2] / image.width, 'bottom': bounds[3] / image.height}
    # API input has a minimum dimension; enlarging is presentation only. All
    # segmentation and geometry still use the original source pixels below.
    scale = max(1., min(280 / min(crop.size), 1568 / max(crop.size)))
    resized = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.BICUBIC)
    supplied = Image.new('RGB', (max(280, resized.width), max(280, resized.height)), (127, 127, 127))
    offset = ((supplied.width - resized.width) // 2, (supplied.height - resized.height) // 2)
    supplied.paste(resized, offset)
    buf = io.BytesIO(); supplied.save(buf, 'PNG')
    description = ('the horizontal floral/garland relief on the dark frieze near `region`' if relief.get('shape') == 'garland'
                   else 'the sculptural ornament near `region`, including its attached flourishes, not windows or framing')
    questions = {
        'region': {'type': 'box', 'instructions': f'Locate {description} in `photo`. Tight bounds around the complete ornament. Exclude separate neighboring trim and cornices.'},
        'usable': {'type': 'noul', 'instructions': f'Is {description} in `photo` clearly visible, fully inside the crop, and unobscured by trees, wires or other objects?'},
        'contrast': {'type': 'noul', 'instructions': f'Is {description} in `photo` pale/light colored relief against a visibly darker backing, so a light-pixel mask would recover its sculptural outline? A plain pale plaque or painted image does not qualify.'}}
    state = {'photo': 'relief-region', 'region': region,
             'regionCoordinates': 'approximate search rectangle in full photo, normalized 0..1 from top-left'}
    key = detection_key(buf.getvalue(), state, questions)
    cached = cache / (key + '.json')
    if cached.exists(): answers = json.loads(cached.read_text())
    else:
        from regen.jev import ask
        answers = ask(state, questions,
                      assets={'relief-region': {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(buf.getvalue()).decode()}},
                      tag='capture-relief-' + key[:12], retries=2)
        cached.write_text(json.dumps(answers, indent=2))
    audit = {'status': 'unresolved', 'sourceImage': str(source), 'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'sourceCamera': view['image'], 'contextPixels': bounds, 'answers': answers,
             'detectorVersion': VERSION, 'depthMeasured': False}
    if answers['usable']['noul'] < .65 or answers['contrast']['noul'] < .7:
        return None, {**audit, 'reason': 'semantic-visibility-or-contrast-unconfirmed'}
    b = answers['region']['box']
    if not (0 <= b['left'] < b['right'] <= 1 and 0 <= b['top'] < b['bottom'] <= 1):
        return None, {**audit, 'reason': 'invalid-detection-box'}
    px = [max(0, int((b['left'] * supplied.width - offset[0]) / scale)),
          max(0, int((b['top'] * supplied.height - offset[1]) / scale)),
          min(crop.width, int(np.ceil((b['right'] * supplied.width - offset[0]) / scale))),
          min(crop.height, int(np.ceil((b['bottom'] * supplied.height - offset[1]) / scale)))]
    if px[2] <= px[0] or px[3] <= px[1]:
        return None, {**audit, 'reason': 'detected-padding-instead-of-relief'}
    overlap = max(0, min(px[2], bounds[2]) - max(px[0], bounds[0])) * max(0, min(px[3], bounds[3]) - max(px[1], bounds[1]))
    if overlap / ((px[2] - px[0]) * (px[3] - px[1])) < .8:
        return None, {**audit, 'reason': 'detected-a-different-region'}
    if relief.get('shape') != 'garland': px = bounds
    crop2 = image.crop(px); mask, stats = split_light_relief(np.array(crop2), isolated=relief.get('shape') != 'garland')
    audit.update(sourcePixels=px, segmentation=stats)
    if mask is None: return None, {**audit, 'reason': stats['reason']}
    # No fabricated surface detail: mask coordinates trace source pixels. The
    # stored source resolution is the actual limit, however large the mesh grows.
    rect = [(px[0] - pad) / ppm / W, (H - (px[3] - pad) / ppm) / H,
            (px[2] - pad) / ppm / W, (H - (px[1] - pad) / ppm) / H]
    result = {**relief, 'rect': rect, 'capturedOutline': vector_mask(mask),
              'outlineEvidence': {**audit, 'sourcePixelSize': list(crop2.size), 'status': 'proposal-not-silhouette-validated'}}
    overlay = np.asarray(crop2).copy(); overlay[mask] = (overlay[mask] * .4 + np.array([30, 225, 230]) * .6).astype('uint8')
    panel = Image.new('RGB', (crop2.width * 2, crop2.height), 'white')
    panel.paste(crop2); panel.paste(Image.fromarray(overlay), (crop2.width, 0))
    panel.thumbnail((900, 400)); panel.save(cache / (key + '-proof.png'))
    audit.update(status='proposal_needs_review', reason='contrast-is-not-shape-validation',
                 proof=str(cache / (key + '-proof.png')), sourcePixelSize=list(crop2.size))
    return result, audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--area', type=Path, required=True)
    ap.add_argument('--profiles', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True); ap.add_argument('--only', type=Path)
    args = ap.parse_args()
    from regen.reading_provenance import load_registered_sides
    data = json.loads(args.profiles.read_text()); registered = load_registered_sides(args.area / 'regen/sides-reg')
    only = set(json.loads(args.only.read_text())) if args.only else None
    cache = args.audit.parent / 'relief-cache'; cache.mkdir(exist_ok=True, parents=True)
    rows = []
    for profile in data['profiles']:
        if only and profile['match'].get('featureId') not in only: continue
        for i, relief in enumerate(profile.get('reliefs', [])):
            result, audit = capture(profile, relief, args.area, registered, cache)
            rows.append({'profile': profile['id'], 'index': i, **audit, **({'proposal': result} if result is not None else {})})
            profile.setdefault('unresolved', []).append({'kind': 'relief-outline', 'index': i, 'reason': audit['reason']})
            print(profile['id'], i, audit['status'], audit.get('reason', ''), flush=True)
    args.output.write_text(json.dumps(data)); args.audit.write_text(json.dumps(rows, indent=2))
    print(json.dumps({'total': len(rows), 'proposals': sum(r['status'] == 'proposal_needs_review' for r in rows), 'applied': 0}))


if __name__ == '__main__': main()
