"""The facade profiles for a ring build: an area's assembled profiles (regen/assemble.py) with the sides re-read for the
ring (areas/ring/readings, reading v2) in place of theirs. A re-read side keeps the reader's own colours (no photo
re-measurement): the readings judge daylight colour by eye, which the shaded or backlit photographs do not give.
Usage: profiles.py <area dir> <base profiles.json> <readings dir> <out.json> [--keep-low] [--nominal]   (run from work/)
Readings are moved onto the wall where the area's registered cameras put them (regen/remap_readings.py) unless --nominal."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'regen'))
import specs
import reading_provenance as provenance

PANE_FIELDS = {'sash', 'lites', 'mullions', 'transoms', 'meetingRail'}


def _matches_source(actual, expected):
    """Compare a small nested source guard without allowing unrelated replacements."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _matches_source(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_matches_source(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and abs(float(actual) - float(expected)) <= 1e-6
    return actual == expected


def apply_pane_updates(reading, updates, side_id):
    """Apply only source-guarded, typed window pane fields to indexed raw records."""
    applied = 0
    for update in updates or []:
        if not isinstance(update, dict):
            raise ValueError(f'{side_id}: pane update must be an object')
        collection, index = update.get('collection'), update.get('index')
        if collection not in ('banks', 'openings', 'bays'):
            raise ValueError(f'{side_id}: invalid pane collection {collection!r}')
        records = reading.get(collection) or []
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(records):
            raise ValueError(f'{side_id}: invalid pane index {index!r} in {collection}')
        current, expected = records[index], update.get('source')
        if not isinstance(expected, dict) or not expected:
            raise ValueError(f'{side_id}: pane update requires a source guard')
        if not _matches_source(current, expected):
            raise ValueError(f'{side_id}: stale pane source guard at {collection}[{index}]')
        if update.get('confidence') not in ('high', 'medium'):
            raise ValueError(f'{side_id}: pane update requires high/medium confidence')
        targets = []
        for face in ('front', 'side'):
            if face in update:
                if collection != 'bays' or not isinstance(update[face], dict):
                    raise ValueError(f'{side_id}: {face} pane fields are supported only on bays')
                targets.append((current.setdefault(face, {}), update[face]))
        if 'fields' in update:
            if not isinstance(update['fields'], dict):
                raise ValueError(f'{side_id}: pane fields must be an object')
            targets.append((current, update['fields']))
        if not targets:
            raise ValueError(f'{side_id}: pane update has no typed fields')
        for target, fields in targets:
            if not fields or set(fields) - PANE_FIELDS:
                raise ValueError(f'{side_id}: pane update contains unsupported fields {sorted(set(fields) - PANE_FIELDS)}')
            target.update(fields)
        applied += 1
    return applied

area, base, rdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
# Typed component rereads are external evidence, applied before registration/height transforms.
COMPONENTS={}
COMPONENTS_PATH = None
CALIBRATION = None
if '--camera-calibration' in sys.argv:
    CALIBRATION = json.loads(Path(sys.argv[sys.argv.index('--camera-calibration')+1]).read_text())
if '--component-observations' in sys.argv:
    _path=Path(sys.argv[sys.argv.index('--component-observations')+1])
    COMPONENTS_PATH = _path
    COMPONENTS={q['side']:q for q in json.load(open(_path)) if isinstance(q,dict) and q.get('side')}
pieces = {p['id']: p for p in json.load(open(area / 'regen/sides/sides.json'))['pieces']}
registered_directory = (Path(sys.argv[sys.argv.index('--registered-sides')+1])
                        if '--registered-sides' in sys.argv else area / 'regen/sides-reg')
registered_sides = provenance.load_registered_sides(registered_directory)
calibrated_sides = (provenance.load_registered_sides(Path(sys.argv[sys.argv.index('--calibrated-sides')+1]))
                    if '--calibrated-sides' in sys.argv else {})
# readings measured on crops cut from provider camera poses are moved onto the wall where the registered cameras put
# what they measured (regen/remap_readings.py); --nominal keeps them as read
import os
os.environ.setdefault('SFW_AREA', str(area)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
WALLMAP = None
if '--nominal' not in sys.argv and (area / 'imagery/registered-cameras.json').exists():
    import frontage_crops_v15 as fc
    from register_cameras import apply_registration
    from remap_readings import WallMap, remap
    _world, _frame = fc.load_world(); _nom = fc.load_cameras(_frame); _reg = apply_registration(_nom, area / 'imagery/registered-cameras.json', bias_fallback=False)
    # a side whose two registered views disagree by over 1.5 m where the provider's agreed holds one bad registration:
    # its reading stays as read (regen/views_agree.py; the check cannot see a bias two cameras share, so it only vetoes)
    _va = json.load(open(Path(__file__).resolve().parent / 'views-agree.json')) if (Path(__file__).resolve().parent / 'views-agree.json').exists() else None
    def _split(pid):
        if _va is None or pid not in _va['A'] or pid not in _va['B']: return False
        import math as _m
        a, b = _va['A'][pid], _va['B'][pid]
        return min(a['peak'], b['peak']) > 6 and _m.hypot(b['dxM'], b['dyM']) > 1.5 and _m.hypot(a['dxM'], a['dyM']) <= 1.5
    def WALLMAP(r, p):
        if _split(p['id']): return None
        v = (p.get('views') or [None] * 2)[int(r.get('use') or 0)] if p.get('views') else None
        if v is None or v['image'] not in _reg or not _reg[v['image']].get('registered') or v.get('cameraGroundZ') is None: return None
        cn, cr = _nom[v['image']], _reg[v['image']]; gr = fc.ground_at(cr['xy'])
        if gr is None: return None
        return WallMap(p, cn['R'], [*cn['xy'], v['cameraGroundZ'] + fc.EYE_HEIGHT_M], cr['R'], [*cr['xy'], float(gr) + fc.EYE_HEIGHT_M + cr.get('heightOffsetM', 0.)])


def zfix(r, p):
    """A reading rectified squashed (its wall top under the LiDAR wall AND its storeys under 2.6 m apart, both)
    has every height stretched back by the LiDAR wall / its wall top, at most 1.6x (a wall top within 5 % of the LiDAR's is left). One signal alone is not enough: a
    roof structure makes the LiDAR wall taller, and some buildings have low storeys."""
    import numpy as np
    # A calibrated camera already maps photo rays onto the wall. A second
    # storey-height stretch would undo that measured projective transform.
    if r.get('calibratedCrops') or r.get('cameraCalibration', {}).get('status') == 'applied': return r, 1.
    top = r.get('wallTopM'); st = [z for z in (r.get('storeys') or []) if isinstance(z, (int, float))]
    if not isinstance(top, (int, float)) or top <= 0 or len(st) < 4: return r, 1.
    pitch = float(np.median(np.diff(st[1:]))); k = p['heightM'] / top
    if not (top / p['heightM'] < .95 and pitch < 2.6): return r, 1.
    k = float(min(1.6, k)); Z = ('z0', 'z1', 'zM', 'wallTopM', 'openingHeightM', 'thicknessM', 'transomM', 'railingHeightM', 'heightM', 'courseM')
    def walk(o):
        if isinstance(o, dict):
            for key, v in list(o.items()):
                if key in Z and isinstance(v, (int, float)): o[key] = v * k
                elif key in ('storeys', 'levels', 'zCentersM') and isinstance(v, list): o[key] = [z * k if isinstance(z, (int, float)) else z for z in v]
                elif key == 'rows' and isinstance(v, list): o[key] = [[a * k, b * k] for a, b in v if isinstance(a, (int, float))]
                elif key == 'parapetTop' and isinstance(v, list): o[key] = [[x, z * k] for x, z in v]
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    r = json.loads(json.dumps(r)); walk(r); return r, k
prof = json.load(open(base)); byid = {p['id']: i for i, p in enumerate(prof['profiles'])}
done, dropped, moved = [], [], []
material_overrides_applied = 0
ornament_overrides_applied = 0
pane_overrides_applied = 0
# a building whose only readings are low-confidence keeps them (a rough reading beats the generic facade); one with a
# better-read side drops its low ones for the style that side gives the rest of the building
KEEP = set(json.load(open(rdir.parent / 'keep-low.json'))['keep']) if (rdir.parent / 'keep-low.json').exists() else set()   # low readings kept by hand (areas/ring/keep-low.json says why)
# a low reading a reviewer has redone against the model and other cameras (areas/ring/reviews) beats the generic facade
KEEP |= {s['side'] for f in (rdir.parent / 'reviews').glob('*.json') for s in json.load(open(f)).get('sides', []) if s.get('readingChanged')}
good = {pieces[f.stem]['building'] for f in rdir.glob('*.json') if f.stem in pieces
        and (lambda r: r.get('thisBuilding') is not False and r.get('materials') and r.get('confidence') in ('medium', 'high'))(json.load(open(f)))}
REG = rdir.parent / 'readings-reg'      # re-readings made on crops cut from the registered cameras (regen/sides-reg)
for f in sorted(rdir.glob('*.json')):
    rr = REG / f.name; fresh = rr.exists()
    reading_path = rr if fresh else f
    r = json.load(open(reading_path))
    source_reading_file_hash = provenance.sha256_file(reading_path)
    p = pieces.get(r.get('id', f.stem))
    if p is None: print('no side record for', f.stem); continue
    registered_required = bool(fresh or r.get('registeredCrops'))
    source_sides = calibrated_sides if r.get('calibratedCrops') else registered_sides
    source = provenance.select_source_view(p, source_sides, r.get('use', 0), registered_required=registered_required)
    if r.get('calibratedCrops') and not (source.get('view') or {}).get('gravityCalibration'):
        raise ValueError(f'{p["id"]}: calibrated reading requires its corrected crop metadata')
    source_side_path = source.get('side_path')
    if source['status'] == 'ok':
        crop_root = (source_side_path.parent if source_side_path else area / 'regen/sides') / 'img'
        missing_crop_files = provenance.missing_crop_files(source, crop_root)
        if missing_crop_files:
            source = {**source, 'status': 'unresolved', 'reason': 'missing crop assets: ' + ', '.join(missing_crop_files)}
    # Keep the registered crop's camera identity while retaining the already matched
    # nominal geometry. An unresolved source gets a deliberately blank ID below.
    source_piece = dict(p)
    if source['status'] == 'ok':
        source_piece['views'] = p.get('views', [])
        if registered_required:
            source_piece['views'] = source_sides[p['id']]['piece']['views']
    else:
        source_piece['views'] = [{'image': '__unresolved__'}]
        r['use'] = 0
    _components=COMPONENTS.get(p['id'])
    if _components is not None:
        if isinstance(_components.get('ornaments'),list):
            r={**r,'ornaments':_components['ornaments']}
        # A partial ornament refinement changes only the typed semantics of
        # individually matched raw records. Other uncertain ornaments, such as
        # compound friezes, remain byte-for-byte as originally read.
        for override in _components.get('ornamentUpdates') or []:
            if not isinstance(override,dict):
                raise ValueError(f'{p["id"]}: ornament update must be an object')
            index=override.get('ornamentIndex'); ornaments=r.get('ornaments') or []
            if isinstance(index,bool) or not isinstance(index,int) or index < 0 or index >= len(ornaments):
                raise ValueError(f'{p["id"]}: invalid ornamentIndex {index!r}')
            current=ornaments[index]; expected=override.get('source')
            if not isinstance(expected,dict):
                raise ValueError(f'{p["id"]}: ornament update requires a source guard')
            for key in ('x0','x1','z0','z1','hex','projectionM'):
                if key not in expected: continue
                actual,want=current.get(key),expected[key]
                if isinstance(want,(int,float)) and not isinstance(want,bool):
                    if not isinstance(actual,(int,float)) or abs(float(actual)-float(want)) > 1e-6:
                        raise ValueError(f'{p["id"]}: stale ornament source guard at index {index} ({key})')
                elif actual != want:
                    raise ValueError(f'{p["id"]}: stale ornament source guard at index {index} ({key})')
            if override.get('confidence') not in ('high','medium'):
                raise ValueError(f'{p["id"]}: ornament update requires high/medium confidence')
            typed={key:override[key] for key in ('shape','repeatCount','ringCount','curveDepth','outerOutline','motif','cardinalFlourishes') if key in override}
            if not typed:
                raise ValueError(f'{p["id"]}: ornament update has no typed descriptor')
            current.update(typed)
            ornament_overrides_applied += 1
        # External material rereads are refinements of observed raw records, not
        # replacement readings. Match the indexed record and its original bounds,
        # family, and colour before overlaying typed structure/finish fields.
        # This keeps profile generation from silently applying evidence to a
        # shifted or recoloured surface.
        for override in _components.get('materials') or []:
            if not isinstance(override,dict):
                raise ValueError(f'{p["id"]}: material override must be an object')
            index = override.get('materialIndex')
            mats = r.get('materials') or []
            if isinstance(index,bool) or not isinstance(index,int) or index < 0 or index >= len(mats):
                raise ValueError(f'{p["id"]}: invalid materialIndex {index!r}')
            current = mats[index]
            expected = override.get('source')
            if not isinstance(expected,dict):
                raise ValueError(f'{p["id"]}: material override requires a source guard')
            for key in ('x0','x1','z0','z1','family','hex'):
                actual, want = current.get(key), expected.get(key)
                if isinstance(want,(int,float)) and not isinstance(want,bool):
                    if not isinstance(actual,(int,float)) or abs(float(actual)-float(want)) > 1e-6:
                        raise ValueError(f'{p["id"]}: stale material source guard at index {index} ({key})')
                elif actual != want:
                    raise ValueError(f'{p["id"]}: stale material source guard at index {index} ({key})')
            confidence = override.get('confidence')
            if confidence not in ('high','medium'):
                raise ValueError(f'{p["id"]}: material override requires high/medium confidence')
            typed = {key: override[key] for key in ('pattern','finish','moduleM') if key in override}
            if not typed:
                raise ValueError(f'{p["id"]}: material override has no typed descriptor')
            current.update(typed)
            material_overrides_applied += 1
        pane_overrides_applied += apply_pane_updates(r, _components.get('paneUpdates'), p['id'])
    pid = 'regen-' + p['id']
    # a side whose photographs show another wall, or that the reader could barely see (an oblique side wall rectified at
    # the wrong scale, a wall behind a hedge), keeps no reading: the generator gives it the building's measured style
    # from its other sides instead of a guess
    if r.get('thisBuilding') is False or not r.get('materials') or (r.get('confidence') == 'low' and p['building'] in good and p['id'] not in KEEP and '--keep-low' not in sys.argv):
        if pid in byid: prof['profiles'][byid[pid]] = None
        dropped.append(p['id'])
        continue
    if WALLMAP is not None and not registered_required:   # a reading made on registered crops is in place
        wm = WALLMAP(r, p)
        if wm is not None:
            dx = float(wm(p['lengthM'] / 2, 3.)[0]) - p['lengthM'] / 2
            r = remap(r, wm); moved.append((p['id'], round(dx, 2)))
    if CALIBRATION is not None and not r.get('calibratedCrops'):
        from regen.calibration_readings import calibrate_reading
        import frontage_crops_v15 as fc
        r = calibrate_reading(r, p, source.get('view') or {}, CALIBRATION, fc.EYE_HEIGHT_M)
    r, k = zfix(r, p)
    if k != 1.: print(f'{p["id"]}: heights stretched x{k:.2f} (squashed rectification)')
    # a ring reading measures every window, so a bank is separate windows unless it says it is a ribbon (one band of
    # glass a storey): the generator's own guess joined three casements a third of a metre apart into one wide frame
    for o in (r.get('banks') or []) + (r.get('openings') or []): o.setdefault('ribbon', o.get('kind') == 'curtain')
    resulting_reading_hash = provenance.reading_sha256(r)
    new = specs.to_profile(specs.clean(r, p), source_piece)
    new['sourceIds'] = source.get('sourceIds', [])
    if isinstance(new.get('materialEvidence'), dict): new['materialEvidence']['sourceIds'] = new['sourceIds']
    side_path = source_side_path
    chosen_view = source.get('view') or {}
    if source['status'] != 'ok':
        new['sourceIds'] = []
        if isinstance(new.get('materialEvidence'), dict): new['materialEvidence']['sourceIds'] = []
        for opening in new.get('openings', []): opening['sourceIds'] = []
        new.setdefault('unresolved', []).append({'kind': 'reading_provenance', 'reason': source['reason']})
        new['status'] = 'unresolved-reading-provenance'
    else:
        new['status'] = 'image-measured-ring-v2'
    new['readingProvenance'] = {
        'status': source['status'], 'accepted': source['status'] == 'ok', 'imageVerified': False,
        'pathBase': 'area',
        'sourceReading': {'path': provenance.relative_path(reading_path, area), 'sha256': source_reading_file_hash},
        'freshRegisteredReading': fresh, 'registeredCropSource': registered_required,
        'sourceSide': provenance.relative_path(side_path, area) if side_path else ('regen/sides/sides.json' if not registered_required else None),
        'image': chosen_view.get('image'), 'rect': chosen_view.get('rect'), 'persp': chosen_view.get('persp'),
        'componentObservation': ({'path': provenance.relative_path(COMPONENTS_PATH, area), 'sha256': provenance.sha256_file(COMPONENTS_PATH)} if COMPONENTS_PATH else None),
        'resultingReadingSha256': resulting_reading_hash,
        **({'calibratedCropSource': True, 'sourceCalibration': r.get('sourceCalibration')} if r.get('calibratedCrops') else {}),
        **({'cameraCalibration': r['cameraCalibration']} if r.get('cameraCalibration') else {}),
        **({'reason': source['reason']} if source['status'] != 'ok' else {})
    }
    if pid in byid: prof['profiles'][byid[pid]] = new
    else: prof['profiles'].append(new)
    done.append(p['id'])
prof['profiles'] = [p for p in prof['profiles'] if p is not None]
prof['method'] = prof.get('method', '') + f' + ring v2 readings ({rdir.as_posix()})'
json.dump(prof, open(out, 'w'))
print(json.dumps({'reread': len(done), 'dropped': dropped, 'profiles': len(prof['profiles']), 'remapped': len(moved), 'shiftOver1m': sum(abs(d) > 1 for _, d in moved), 'materialOverridesApplied': material_overrides_applied, 'ornamentOverridesApplied': ornament_overrides_applied, 'paneOverridesApplied': pane_overrides_applied}))
