"""Viewer camera poses that reproduce the perspective photographs of the buildings round a park, so a render of the
model can stand beside each photograph at the same viewpoint (same camera position, direction and field of view as
regen/sides.py used to cut the photograph's _persp crop from its panorama).
Usage: poses.py <area dir> <park OSM way id> <world.json> <out.json> [--within 32] [--views 2] [--all-sides]
                [--registered <area>/imagery/registered-cameras.json]   (SFW_AREA set to the area; run from work/)
With --registered, use the registered position and height and the world-forward vector used to create that registered
image crop. Panorama yaw/tilt already participates in crop creation; applying it again to world-forward is incorrect."""
import argparse, glob, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
from regen.reading_provenance import load_registered_sides, select_source_view


def matching_perspective(area, side_id, image_id, nominal_view, registered_views, registered, source_directory=None):
    """Return the image generated from the same side/camera state as a saved pose."""
    if registered:
        view = registered_views.get((side_id, image_id))
        if view is None or 'registered' not in view:
            raise RuntimeError(f"no registered perspective metadata for {side_id} camera {image_id}")
        image_path = (Path(source_directory) if source_directory else Path(area) / 'regen/sides-reg') / 'img' / view['persp']
    else:
        view = nominal_view
        image_path = Path(area) / 'regen/sides/img' / view['persp']
    if not image_path.is_file():
        raise FileNotFoundError(f"pose image does not exist: {image_path}")
    return image_path, view


def vfov_for_image(hfov_deg, width, height):
    """Recover vertical FOV from the saved perspective crop aspect and its horizontal FOV."""
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * height / width))


def side_camera_geometry(side):
    """Reconstruct the wall input used by sides.py from a side metadata record."""
    normal = np.asarray(side['normal'], float)
    tangent = np.array([-normal[1], normal[0]])
    midpoint = normal * side['planeV'] + tangent * sum(side['frameU']) / 2
    return {'midpoint': midpoint, 'normal': normal, 'lengthM': side['lengthM'],
            'heightM': side['heightM']}


def pose_source_views(nominal_side, registered_sides, registered):
    """Choose the side record and views that actually generated pose crops."""
    if not registered:
        views = nominal_side.get('views') or []
        return {'status': 'ok' if views else 'unresolved', 'piece': nominal_side, 'views': views,
                'reason': None if views else 'nominal side has no views', 'unresolvedViews': []}
    side_id = nominal_side.get('id')
    source = registered_sides.get(side_id)
    if source is None:
        return {'status': 'unresolved', 'reason': 'registered side record missing', 'views': [], 'unresolvedViews': []}
    if source.get('status') != 'ok':
        return {'status': 'unresolved', 'reason': 'registered side record ambiguous', 'views': [],
                'unresolvedViews': [str(path) for path in source.get('candidates', [])]}
    registered_side = source['piece']
    raw_views = registered_side.get('views') or []
    if not raw_views:
        return {'status': 'unresolved', 'reason': 'registered side has no views', 'views': [], 'unresolvedViews': []}
    views, unresolved = [], []
    for index, view in enumerate(raw_views):
        selected = select_source_view(nominal_side, registered_sides, index, registered_required=True)
        if selected['status'] != 'ok':
            unresolved.append({'view': index, 'reason': selected['reason']})
            continue
        if 'registered' not in view:
            unresolved.append({'view': index, 'image': view.get('image'), 'reason': 'registered crop metadata missing registered marker'})
            continue
        views.append({**view, '_sourceViewIndex': index})
    if not views:
        reason = unresolved[0]['reason'] if unresolved else 'registered side has no usable views'
        return {'status': 'unresolved', 'reason': reason, 'views': [], 'unresolvedViews': unresolved}
    return {'status': 'ok', 'piece': registered_side, 'views': views, 'reason': None, 'unresolvedViews': unresolved}


def resolve_camera_state(view, camera, accepted_registrations, bias, eye_height):
    """Reproduce the placement mode recorded on a registered side crop."""
    if 'registered' not in view:
        return {'status': 'unresolved', 'reason': 'registered camera state missing'}
    marker = view['registered']
    registration = accepted_registrations.get(view.get('image'))
    nominal_xy = np.asarray(camera['xy'], float)
    if marker == 'skyline':
        if registration is None:
            return {'status': 'unresolved', 'reason': 'crop says skyline-registered but accepted camera registration is missing'}
        return {'status': 'ok', 'xy': np.asarray(registration['xy'], float),
                'cameraHeightM': registration['cameraHeightM'], 'registration': registration, 'biasOnly': False}
    if marker == 'bias':
        if registration is not None:
            return {'status': 'unresolved', 'reason': 'crop says bias-placed but camera has an accepted skyline registration'}
        if bias is None:
            return {'status': 'unresolved', 'reason': 'crop says bias-placed but shared camera bias is unavailable'}
        return {'status': 'ok', 'xy': nominal_xy + np.asarray(bias, float),
                'cameraHeightM': eye_height, 'registration': None, 'biasOnly': True}
    if marker is False:
        if registration is not None:
            return {'status': 'unresolved', 'reason': 'crop says nominal-placed but camera has an accepted skyline registration'}
        return {'status': 'ok', 'xy': nominal_xy, 'cameraHeightM': eye_height,
                'registration': None, 'biasOnly': False}
    return {'status': 'unresolved', 'reason': f'unsupported registered camera state {marker!r}'}


def main():
    sys.path.insert(0, '.')
    import frontage_crops_v15 as fc

    ap = argparse.ArgumentParser(); ap.add_argument('area'); ap.add_argument('park'); ap.add_argument('world'); ap.add_argument('out')
    ap.add_argument('--within', type=float, default=32.); ap.add_argument('--views', type=int, default=2); ap.add_argument('--all-sides', action='store_true', help='every photographed side, not only the one facing the park'); ap.add_argument('--registered')
    ap.add_argument('--source-sides', type=Path, help='registered/corrected metadata and img directory used for comparison')
    ap.add_argument('--only', type=Path, help='explicit source-defined review cohort, instead of park-distance selection')
    a = ap.parse_args()
    if a.source_sides and not a.registered:
        ap.error('--source-sides requires --registered camera placement')
    reg = {k: r for k, r in json.load(open(a.registered)).items() if r.get('accepted')} if a.registered else {}
    # a camera the skyline could not place still carries the provider's shared bias: it moves by the registered cameras' median
    bias = np.median([[r['dx'], r['dy']] for r in reg.values()], 0) if len(reg) >= 10 else None
    area = Path(a.area); world, frame = fc.load_world(); cams = {k: c for k, c in fc.load_cameras(frame).items() if c['exists']}
    root = ET.parse(fc.OSM_PATH).getroot(); nodes = {n.get('id'): frame.local(float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
    park = next(Polygon([tuple(nodes[nd.get('ref')]) for nd in w.iter('nd')]) for w in root.iter('way') if w.get('id') == a.park)
    centre = np.array(park.centroid.coords[0]); fps = fc.load_footprints(frame)
    built = {b['id'] for b in json.load(open(a.world))['buildings']}
    ring = sorted({b for b in built if b.split('_p')[0] in fps and fps[b.split('_p')[0]].distance(park) < a.within})
    if a.only:
        selected = json.loads(a.only.read_text())
        if not isinstance(selected, list) or not all(isinstance(q, str) for q in selected):
            ap.error('--only must be a JSON list of building identities')
        ring = sorted(set(selected) & built)
    pieces = {}
    for f in sorted(glob.glob(str(area / 'regen/sides/sides-*.json'))):
        for p in json.load(open(f))['pieces']:
            if p['views'] and (p['id'] not in pieces or len(p['views']) > len(pieces[p['id']]['views'])): pieces[p['id']] = p
    source_directory = a.source_sides or area / 'regen/sides-reg'
    registered_sides = load_registered_sides(source_directory) if a.registered else {}
    out = []
    unresolved = []
    for bid in ring:
        ps = [p for p in pieces.values() if p['building'] == bid]
        for p in ps:
            n = np.asarray(p['normal'], float); t = np.array([-n[1], n[0]]); p['_mid'] = n * p['planeV'] + t * sum(p['frameU']) / 2
            d = centre - p['_mid']; p['_face'] = float(n @ (d / np.linalg.norm(d)))
        # the side that faces the park (else the longest street side), its views in the order regen/sides.py ranked them
        ps.sort(key=lambda p: (-(p['_face'] > .3), -p['streetFacing'], -p['lengthM']))
        for p in (ps if a.all_sides else ps[:1]):
            source = pose_source_views(p, registered_sides, bool(a.registered))
            if source['status'] != 'ok':
                unresolved.append({'building': bid, 'side': p['id'], 'registered': bool(a.registered),
                                   'status': 'unresolved', 'reason': source['reason'],
                                   'candidates': source.get('unresolvedViews', [])})
                continue
            for item in source.get('unresolvedViews', []):
                unresolved.append({'building': bid, 'side': p['id'], 'registered': bool(a.registered),
                                   'status': 'unresolved', **item})
            camera_side = source['piece']
            for k, v in enumerate(source['views'][:a.views]):
                if v['image'] not in cams:
                    unresolved.append({'building': bid, 'side': p['id'], 'view': k, 'image': v['image'],
                                       'registered': bool(a.registered), 'status': 'unresolved',
                                       'reason': 'camera record missing'})
                    continue
                cam = cams[v['image']]
                camera_state = resolve_camera_state(v, cam, reg, bias, fc.EYE_HEIGHT_M) if a.registered else {
                    'status': 'ok', 'xy': np.asarray(cam['xy'], float), 'cameraHeightM': fc.EYE_HEIGHT_M,
                    'registration': None, 'biasOnly': False}
                if camera_state['status'] != 'ok':
                    unresolved.append({'building': bid, 'side': p['id'], 'view': v.get('_sourceViewIndex', k),
                                       'image': v['image'], 'registered': bool(a.registered),
                                       'status': 'unresolved', 'reason': camera_state['reason']})
                    continue
                r = camera_state['registration']
                view_xy = camera_state['xy']
                # The registered crop was generated from the registered side's geometry and camera position.
                # Keep the side's world-forward direction: apply_registration changes R, which maps panorama rays
                # to world space, but render_perspective receives the world-forward vector independently.
                piece = side_camera_geometry(camera_side)
                vp = fc.view_parameters(view_xy, piece, camera_side.get('groundZ', p['groundZ']))
                hf = min(100, vp['hfovDeg'] * 1.25)
                vf_crop = min(100, vp['vfovDeg'] * 1.15)
                fx = 500 / math.tan(math.radians(hf) / 2); expected_h = int(round(2 * fx * math.tan(math.radians(vf_crop) / 2)))
                if expected_h > 1000:
                    expected_h = 1000
                    vf_crop = math.degrees(2 * math.atan(500 / fx))
                # the eye stands on the camera's own ground: on these hills the street under a camera is often metres above or below
                # the side's ground line, and a render from the side's ground sees the building from the wrong height
                eye = [float(view_xy[0]), float(view_xy[1]), float(fc.ground_at(view_xy)) + camera_state['cameraHeightM']]; f = np.asarray(vp['forward'], float)
                image_error = None
                try:
                    image_path, _ = matching_perspective(area, p['id'], v['image'], v,
                                                         {(p['id'], v['image']): v}, bool(a.registered), source_directory)
                except (RuntimeError, FileNotFoundError) as exc:
                    image_path = None
                    image_error = str(exc)
                if image_path is not None:
                    with Image.open(image_path) as image:
                        image_width, h = image.size
                    vf = vfov_for_image(hf, image_width, h)
                elif a.registered:
                    unresolved.append({'building': bid, 'side': p['id'], 'view': v.get('_sourceViewIndex', k),
                                       'image': v['image'], 'registered': bool(a.registered),
                                       'status': 'unresolved', 'reason': image_error})
                    continue
                else:
                    image_width, h, vf = 1000, expected_h, vf_crop
                out.append({'registered': 'bias' if camera_state['biasOnly'] else bool(r), 'building': bid, 'side': p['id'], 'facesPark': p['_face'] > .3, 'view': v.get('_sourceViewIndex', k), 'image': v['image'], 'captureDate': v.get('captureDate'),
                        'persp': str(image_path) if image_path is not None else None, 'imageAvailable': image_path is not None,
                        'imageError': image_error, 'from': [round(x, 3) for x in eye],
                        'at': [round(e + 20 * x, 3) for e, x in zip(eye, f)], 'vfovDeg': round(vf, 3), 'width': image_width, 'height': h})
    document = {'park': a.park, 'ring': ring, 'poses': out}
    if a.registered:
        document['unresolved'] = unresolved
    json.dump(document, open(a.out, 'w'), indent=1)
    print(json.dumps({'ring': len(ring), 'withPhotos': len({o['building'] for o in out}), 'poses': len(out), 'unresolved': len(unresolved), 'noPhotos': [b for b in ring if b not in {o['building'] for o in out}]}))


if __name__ == '__main__':
    main()
