"""Compile accepted paired facade-course measurements into derived profiles.

This records measured surface evidence only. It does not change facade color,
geometry, or the source profiles; downstream renderers may use moduleEvidence
after review. Run from work/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'areas/review/facade-rethink/paired-local-lettering-v5/profiles.json'
REPORT = ROOT / 'areas/review/facade-rethink/horizontal-module-audit-r2/report.json'
SEMANTIC = ROOT / 'areas/review/facade-rethink/horizontal-module-audit-r2/semantic-review.json'
READINGS = ROOT / 'areas/review/facade-rethink/readings-recovered-v16'
AREA = ROOT / 'areas/pacific-heights'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compile_profiles(base_path: Path, report_path: Path, semantic_path: Path,
                     output_path: Path, area: Path = AREA,
                     readings_dir: Path = READINGS) -> dict:
    profiles_doc = json.loads(base_path.read_text())
    profiles = {p['id']: p for p in profiles_doc['profiles']}
    semantic = {r['profileId']: r for r in json.loads(semantic_path.read_text())['records']
                if r.get('accepted')}
    measured = {r['profileId']: r for r in json.loads(report_path.read_text())['records']
                if r.get('accepted') and r['profileId'] in semantic}
    compiled = []
    for pid, audit in sorted(measured.items()):
        if pid not in semantic:
            raise ValueError(f'{pid}: accepted measurement lacks semantic review')
        profile = profiles.get(pid)
        if profile is None:
            raise ValueError(f'{pid}: missing source profile')
        evidence = profile.get('materialEvidence') or {}
        if evidence.get('family') != 'timber' or not profile.get('material'):
            raise ValueError(f'{pid}: primary wall material is not identified as timber')
        prov = profile.get('readingProvenance') or {}
        src = prov.get('sourceReading') or {}
        if prov.get('status') != 'ok' or not prov.get('accepted') or not prov.get('registeredCropSource'):
            raise ValueError(f'{pid}: source provenance is not accepted/registered')
        reading_path = readings_dir / Path(src.get('path', '')).name
        if not reading_path.is_file() or sha(reading_path) != src.get('sha256'):
            raise ValueError(f'{pid}: source reading missing or hash mismatch')
        reading = json.loads(reading_path.read_text())
        color = evidence.get('colorRgb')
        material_hex = '#' + ''.join(f'{int(c):02x}' for c in color) if color else None
        zones = [z for z in reading.get('materials', [])
                 if z.get('family') == 'timber' and str(z.get('hex', '')).lower() == str(material_hex).lower()]
        if not zones:
            raise ValueError(f'{pid}: no reading zone owns the profile primary material {material_hex}')
        # Select the widest matching measured zone, which is the dominant wall
        # surface in these accepted records; keep its bounds attached to the evidence.
        zone = max(zones, key=lambda z: (float(z['x1']) - float(z['x0'])) *
                   (float(z['z1']) - float(z['z0'])))
        sem = semantic[pid]
        views = sem.get('semanticViews', [])
        if len(views) < 2 or any(not v.get('accepted') or
                                 not (v.get('verifiedVertical') or {}).get('accepted') or
                                 (v.get('verifiedHorizontalControl') or {}).get('accepted')
                                 for v in views[:2]):
            raise ValueError(f'{pid}: paired vertical repeat / horizontal-axis control gate failed')
        pitches = [float(v['verifiedVertical']['period_m']) for v in views[:2]]
        if abs(pitches[0] - pitches[1]) / max(pitches) > .05:
            raise ValueError(f'{pid}: paired measured pitches disagree by over 5%')
        source_views = [p.get('sourceImageId') for p in audit.get('patches', [])[:2]]
        if len(source_views) != 2 or source_views[0] == source_views[1]:
            raise ValueError(f'{pid}: paired registered source images are missing')
        crop_hashes = []
        for patch in audit['patches'][:2]:
            crop = (ROOT / patch['rectPath']).resolve()
            if not crop.is_file():
                raise ValueError(f'{pid}: registered crop missing: {crop}')
            crop_hashes.append({'imageId': patch['sourceImageId'], 'sha256': sha(crop)})
        profile['moduleEvidence'] = {
            'status': 'paired-semantic-review-accepted',
            'pattern': 'horizontal_periodic_boards',
            'axis': 'horizontal_courses',
            'pitchM': round(sum(pitches) / len(pitches), 5),
            'pitchByViewM': [round(x, 5) for x in pitches],
            'sourceZoneM': {k: float(zone[k]) for k in ('x0', 'x1', 'z0', 'z1')},
            'material': {'family': 'timber', 'hex': material_hex,
                         'profileMaterial': profile['material']},
            'sourceReading': {'path': src['path'], 'sha256': src['sha256']},
            'pairedRegisteredCrops': crop_hashes,
            'audit': {'report': str(report_path.relative_to(ROOT)),
                      'semanticReview': str(semantic_path.relative_to(ROOT)),
                      'coursePitchM': audit.get('coursePitchM'),
                      'crossViewAgreement': sem.get('agreement')},
            'scope': 'measured material zone only; no building-wide recolor or geometry inferred'}
        compiled.append({'profileId': pid, 'featureId': audit.get('featureId'),
                         'pitchM': profile['moduleEvidence']['pitchM'],
                         'materialHex': material_hex,
                         'zoneM': profile['moduleEvidence']['sourceZoneM']})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles_doc, indent=2) + '\n')
    return {'schema': 'compiled-horizontal-modules-v1', 'baseProfiles': str(base_path),
            'baseSha256': sha(base_path), 'compiledProfiles': str(output_path),
            'count': len(compiled), 'records': compiled,
            'limitations': ['evidence is attached to derived profiles only',
                            'visibility depends on whether the source zone intersects a rendered wall cell',
                            'module evidence must remain clipped to sourceZoneM']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profiles', type=Path, default=BASE)
    ap.add_argument('--report', type=Path, default=REPORT)
    ap.add_argument('--semantic-review', type=Path, default=SEMANTIC)
    ap.add_argument('--area', type=Path, default=AREA)
    ap.add_argument('--readings-dir', type=Path, default=READINGS)
    ap.add_argument('--output', type=Path, default=ROOT / 'areas/review/facade-rethink/horizontal-module-audit-r2/release/profiles.json')
    a = ap.parse_args()
    result = compile_profiles(a.profiles, a.report, a.semantic_review, a.output, a.area, a.readings_dir)
    manifest = a.output.parent / 'manifest.json'
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'count': result['count'], 'profiles': result['records'],
                      'output': str(a.output), 'manifest': str(manifest)}, indent=2))


if __name__ == '__main__':
    main()
