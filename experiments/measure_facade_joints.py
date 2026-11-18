"""Bounded photo-to-joint candidate, overlay and reusable material swatch.

Run from the repository: PYTHONPATH=. python experiments/measure_facade_joints.py --help
No world, source image or material observation is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from general_world.facade_joint_grid import measure_joint_grid


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('image', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--crop', nargs=4, type=int, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    selection = cli.add_mutually_exclusive_group(required=True)
    selection.add_argument('--mask', type=Path, help='Same dimensions as original image; nonzero=surface')
    selection.add_argument('--surface-rgb', nargs=3, type=int,
                           help='Color-based candidate mask; not a substrate classification')
    cli.add_argument('--color-tolerance', type=float, default=65.)
    cli.add_argument('--min-pitch-px', type=int, default=18)
    cli.add_argument('--max-pitch-px', type=int, default=140)
    cli.add_argument('--width-m', type=float, help='External crop width; creates PROVISIONAL metric swatch')
    cli.add_argument('--height-m', type=float, help='External crop height; creates PROVISIONAL metric swatch')
    args = cli.parse_args()
    started = time.monotonic()
    if bool(args.width_m) != bool(args.height_m):
        cli.error('supply both --width-m and --height-m, or neither')
    if args.width_m is not None and min(args.width_m, args.height_m) <= 0:
        cli.error('physical extents must be positive')
    # Decode only this source and cap processing size explicitly.
    with Image.open(args.image) as original:
        original_size = original.size
        crop = args.crop or [0, 0, *original.size]
        if not (0 <= crop[0] < crop[2] <= original.width and
                0 <= crop[1] < crop[3] <= original.height):
            cli.error('crop lies outside source image')
        im = original.crop(crop).convert('RGB')
    im.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    rgb = np.array(im)
    if args.mask:
        with Image.open(args.mask) as mask_image:
            if mask_image.size != original_size:
                cli.error('mask must match original image dimensions')
            mask = np.array(mask_image.crop(crop).convert('L').resize(im.size, Image.Resampling.NEAREST)) > 0
        mask_source = {'path': str(args.mask), 'kind': 'external surface observation'}
    else:
        delta = rgb.astype(float) - np.array(args.surface_rgb)
        mask = np.linalg.norm(delta, axis=2) <= args.color_tolerance
        mask_source = {'kind': 'provisional color-distance selection',
                       'rgb': args.surface_rgb, 'tolerance': args.color_tolerance,
                       'substrate_classified': False}
    result = measure_joint_grid(rgb, mask, min_pitch_px=args.min_pitch_px,
                                max_pitch_px=args.max_pitch_px)
    result['source'] = {'path': str(args.image),
                        'sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
                        'crop_original_px': crop, 'processing_size_px': list(im.size),
                        'mask': mask_source}
    args.output.mkdir(parents=True, exist_ok=True)
    # Overlay only supported surface pixels, never project lines over glass/trees.
    overlay = rgb.copy()
    y, x = np.indices(mask.shape)
    for name, axis, other in [('vertical', x, y), ('horizontal', y, x)]:
        fit = result[name]
        if not fit['accepted']:
            continue
        other_extent = im.height if name == 'vertical' else im.width
        q = (axis - fit['shear'] * other) / (1 + fit['fan'] * other / other_extent)
        distance = np.abs((q - fit['phase_px'] + fit['pitch_px']/2)
                          % fit['pitch_px'] - fit['pitch_px']/2)
        hit = (distance <= 1.2) & mask
        overlay[hit] = [24, 210, 150] if name == 'vertical' else [255, 180, 40]
    Image.fromarray(np.uint8(mask) * 255).save(args.output / 'surface-mask.png')
    Image.fromarray(overlay).save(args.output / 'joint-overlay.png')
    # A texture-space demonstration remains separate from world assignment.
    # Pigment comes from selected pixels; no photograph is shipped as a map.
    if mask.any():
        pigment = np.median(rgb[mask], axis=0)
        swatch = np.broadcast_to(pigment, (512, 512, 3)).copy()
        yy, xx = np.indices((512, 512))
        for name, axis in [('vertical', xx), ('horizontal', yy)]:
            fit = result[name]
            if fit['accepted']:
                # Preserve observed number of modules across this crop.
                source_axis = im.width if name == 'vertical' else im.height
                pitch = fit['pitch_px'] / source_axis * 512
                swatch[np.minimum(axis % pitch, pitch - axis % pitch) < 1.1] *= .65
        Image.fromarray(np.uint8(swatch)).save(args.output / 'module-layout.png')
    if args.width_m:
        result['provisional_metric'] = {'crop_extents_m': [args.width_m, args.height_m],
            'verified': False, 'reason': 'external extents; camera registration unverified',
            'vertical_pitch_m': result['vertical'].get('pitch_px', 0) / im.width * args.width_m
                if result['vertical']['accepted'] else None,
            'horizontal_pitch_m': result['horizontal'].get('pitch_px', 0) / im.height * args.height_m
                if result['horizontal']['accepted'] else None}
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_mib'] = round(peak / (1024**2 if sys.platform == 'darwin' else 1024), 2)
    (args.output / 'measurement.json').write_text(json.dumps(result, indent=2) + '\n')
    # Compact review sheet with explicit acceptance on each axis.
    w, h = im.size
    sheet = Image.new('RGB', (w * 2, h + 96), '#171d25')
    sheet.paste(im, (0, 96)); sheet.paste(Image.fromarray(overlay), (w, 96))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), 'SOURCE PHOTO | AUTOMATIC JOINT CANDIDATE', fill='white')
    status = []
    for key in ('vertical', 'horizontal'):
        fit = result[key]
        status.append(f"{key}: {fit['pitch_px']:.1f}px, checked on held-out strips"
                      if fit['accepted'] else f'{key}: no accepted repeat')
    draw.text((16, 36), ' / '.join(status), fill='#9ee5cc')
    draw.text((16, 60), 'Image-space pattern only. Material identity, metric scale and world placement remain unverified.', fill='#ccd1d8')
    sheet.save(args.output / 'review.png')
    print(json.dumps({'output': str(args.output), 'vertical': result['vertical'],
                      'horizontal': result['horizontal'], 'seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
