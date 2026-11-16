"""Measure panorama gravity from image lines; preserve cameras and original crops.

Produces an external calibration manifest and new rectified/perspective evidence.
It never levels an image merely because its metadata is tilted. Unaccepted line
consensus leaves the input camera unchanged. Run from work/ with PYTHONPATH=.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from regen.panorama_gravity import estimate_panorama_gravity, corrected_rotation
from regen.reading_provenance import load_registered_sides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--area', type=Path, required=True)
    ap.add_argument('--only', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    os.environ['SFW_AREA'] = str(a.area.resolve())
    import frontage_crops_v15 as fc
    import auto_read_frontages_v24 as ar
    from regen.register_cameras import apply_registration
    from areas.ring.poses import side_camera_geometry
    from panorama_balance import road_gain
    _, frame = fc.load_world()
    cameras = apply_registration(fc.load_cameras(frame), a.area / 'imagery/registered-cameras.json')
    ids = {q.split('_p')[0] for q in json.loads(a.only.read_text())}
    sides = [s['piece'] for s in load_registered_sides(a.area / 'regen/sides-reg').values()
             if s.get('status') == 'ok' and s['piece']['building'].split('_p')[0] in ids]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'img').mkdir(exist_ok=True)
    path = a.output / 'calibration.json'
    report = json.loads(path.read_text()) if path.exists() else {
        'version': 'image-gravity-v1', 'images': {},
        'limitations': ['Gravity corrects roll/pitch only; location, heading and facade depth remain uncertain.']}
    # Source images alone are not a valid cache key after a heuristic change.
    # Never silently reuse observations produced by an older estimator.
    import cv2
    estimator_hash = hashlib.sha256(
        Path(__file__).with_name('panorama_gravity.py').read_bytes()
        + Path(__file__).read_bytes() + cv2.__version__.encode()).hexdigest()
    report['estimatorSha256'] = estimator_hash
    image_ids = sorted({v['image'] for p in sides for v in p.get('views', [])})
    for k, identity in enumerate(image_ids):
        if identity not in cameras or not cameras[identity]['exists']:
            continue
        camera = cameras[identity]
        digest = hashlib.sha256(camera['path'].read_bytes()).hexdigest()
        cached = report['images'].get(identity)
        if (cached and cached.get('estimatorSha256') == estimator_hash
                and cached.get('sourceSha256') == digest
                and np.allclose(cached['referenceRotation'], camera['R'])):
            continue
        result = estimate_panorama_gravity(fc.load_pano(camera['path']), prior=camera['R'][:, 2])
        gravity = result['gravity'] if result['gravity'] is not None else camera['R'][:, 2]
        proposed = corrected_rotation(camera['R'], gravity)
        delta = float(np.degrees(np.arccos(np.clip(np.dot(camera['R'][:, 2], gravity), -1., 1.))))
        applied = bool(result['accepted'] and delta > .25)
        rotation = proposed if applied else camera['R']
        report['images'][identity] = {
            'sourceSha256': digest, 'sourcePath': str(camera['path']), 'estimatorSha256': estimator_hash,
            'referenceRotation': camera['R'].tolist(), 'rotation': rotation.tolist(),
            'cameraXY': camera['xy'].tolist(), 'registration': camera.get('registered', False),
            'applied': applied, 'correctionDeg': delta, 'estimate': result}
        path.write_text(json.dumps(report, indent=2))
        print(k+1, len(image_ids), identity, 'applied' if applied else 'unchanged', round(delta, 2), flush=True)
    ar.PPM = 45.
    output_sides = []
    for p in sides:
        q = json.loads(json.dumps(p))
        wall = side_camera_geometry(p)
        for v in q.get('views', []):
            result = report['images'].get(v['image'])
            if result is None:
                continue
            camera = cameras[v['image']]
            if v.get('registered', False) != camera.get('registered', False):
                raise ValueError('source crop camera state changed; regenerate source packet before calibration')
            camera = {**camera, 'R': np.asarray(result['rotation']), 'groundZ': v['cameraGroundZ']}
            pano = np.asarray(fc.load_pano(camera['path']), np.float32)
            pano = np.clip(pano * road_gain(pano), 0, 255).astype(np.uint8)
            rect, _, _ = ar.rectify(wall, camera, p['groundZ'], pano)
            Image.fromarray(np.clip(rect * 255, 0, 255).astype(np.uint8)).save(a.output / 'img' / v['rect'])
            vp = fc.view_parameters(camera['xy'], wall, p['groundZ'])
            persp, _ = fc.render_perspective(pano, camera['R'], np.asarray(vp['forward']),
                min(100., vp['hfovDeg'] * 1.25), min(100., vp['vfovDeg'] * 1.15), width=1000, max_height=1000)
            Image.fromarray(persp).save(a.output / 'img' / v['persp'])
            v.update(rectPad=0, ppm=ar.PPM, gravityCalibration={
                'manifest': str(path.resolve()), 'applied': result['applied'],
                'correctionDeg': result['correctionDeg']})
        output_sides.append(q)
    (a.output / 'sides.json').write_text(json.dumps({'pieces': output_sides}, indent=2))
    print(json.dumps({'images': len(report['images']), 'corrected': sum(r['applied'] for r in report['images'].values()),
                      'sides': len(output_sides), 'output': str(a.output)}))


if __name__ == '__main__': main()
