"""Final readings -> the profile file the generator takes (--regen-profiles). Readings measured on crops cut from a
provider camera pose are moved onto the wall where the area's registered camera (regen/register_cameras.py, the 360-degree
skyline against the LiDAR) puts what they measured (regen/remap_readings.py). Measured on the Lafayette Park ring
(2026-10-08): windows land where the photographs have them more often (dark-feature overlap with the photographs, median
0.32 -> 0.38, better in 14 of 26 views, worse in 3), doors stand on the terrain no worse. A side whose two registered
views disagree by over 1.5 m where its provider views agreed (regen/views_agree.py) keeps its reading as read.
Usage: assemble.py [readings dir name, default final] [--nominal]"""
import glob, json, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import specs, remap, colours, aligncorr
import frontage_crops_v15 as fc
# the skyline registrations moved readings by metres where the photographs already matched the walls (a side's windows
# came out 2.5 m left and 1.8 m low): off until a registration is checked against the wall it is used for
USE_REGISTRATION = False      # the old per-panorama registration (registration*.json, remap.py): superseded below
root = pathlib.Path(os.environ.get('REGEN_ROOT', 'regen/v1')); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
reg = {}
# sides built of tier walls but rectified from the street (tier_sides.py measures them from their own base)
RETIRED = set(json.load(open(root / 'retired-tier-sides.json'))) if (root / 'retired-tier-sides.json').exists() else set()
ALIGN = json.load(open(root / 'alignment.json')) if (root / 'alignment.json').exists() else {}
for f in glob.glob(str(root / 'registration*.json')): reg.update(json.load(open(f)))
world, frame = fc.load_world(); cams = fc.load_cameras(frame); bz = {b['id']: b['center'][2] for b in world['buildings']}
AREA = pathlib.Path(os.environ['SFW_AREA']) if os.environ.get('SFW_AREA') else None
REGCAMS = None
if '--nominal' not in sys.argv and AREA is not None and (AREA / 'imagery/registered-cameras.json').exists():
    from register_cameras import apply_registration
    from remap_readings import WallMap, remap as remap_reading
    REGCAMS = apply_registration(cams, AREA / 'imagery/registered-cameras.json', bias_fallback=False)
VA = json.load(open(root / 'views-agree.json')) if (root / 'views-agree.json').exists() else None
def split_views(pid):
    if VA is None or pid not in VA['A'] or pid not in VA['B']: return False
    a, b = VA['A'][pid], VA['B'][pid]
    return min(a['peak'], b['peak']) > 6 and (b['dxM'] ** 2 + b['dyM'] ** 2) ** .5 > 1.5 and (a['dxM'] ** 2 + a['dyM'] ** 2) ** .5 <= 1.5
profiles, stats = [], {'profiles': 0, 'registered': 0, 'skipped': 0}
SRC = sys.argv[1] if len(sys.argv) > 1 else 'final'
# The skyline corrections are first checked against the readings' own ground lines (street-level doors): an alignment
# that moves the ground away from the photographed doors more often than toward them is not applied. Both areas failed
# (10-05: nearer in 1-22% of readings, the ground put 1-1.6 m up); cornices and bays standing out over the street
# likely raise the photographed skyline above the one the LiDAR footprints predict.
n_test, agree = aligncorr.ground_agreement([(f.stem, json.load(open(f))) for f in sorted((root / SRC).glob('*.json'))], ALIGN)
stats['alignmentGroundCheck'] = {'readings': n_test, 'nearerDoors': agree}
if ALIGN and n_test >= 20 and agree < .5: print(f'alignment not applied: its corrections move the ground toward the doors in {agree:.0%} of {n_test} readings'); ALIGN = {}
for f in sorted((root / SRC).glob('*.json')):
    r = json.load(open(f)); p = byid[f.stem]
    if f.stem in RETIRED: stats['retired'] = stats.get('retired', 0) + 1; continue
    if r.get('thisBuilding') is False or not r.get('materials'): stats['skipped'] += 1; continue
    r = specs.clean(r, p); img = p['views'][r['use']]['image']; v = p['views'][r['use']]
    r = colours.measure(r, p, root / 'sides/img')        # on the crop the reading was measured on, at its positions
    moved = False
    if REGCAMS is not None and REGCAMS[img].get('registered') and v.get('cameraGroundZ') is not None and not split_views(f.stem) and not r.get('registeredCrops'):
        cn, cr = cams[img], REGCAMS[img]; gr = fc.ground_at(cr['xy'])
        if gr is not None:
            r = remap_reading(r, WallMap(p, cn['R'], [*cn['xy'], v['cameraGroundZ'] + fc.EYE_HEIGHT_M], cr['R'], [*cr['xy'], float(gr) + fc.EYE_HEIGHT_M + cr.get('heightOffsetM', 0.)]))
            moved = True; stats['remapped'] = stats.get('remapped', 0) + 1
    if not moved: r, c = aligncorr.correct(r, p, ALIGN.get(f.stem)); stats['aligned'] = stats.get('aligned', 0) + bool(c)
    if USE_REGISTRATION and remap.good(reg.get(img)):
        r = remap.remap_reading(r, remap.Map(p, cams[img], reg[img], bz[p['building']])); stats['registered'] += 1
    profiles.append(specs.to_profile(r, p)); stats['profiles'] += 1
json.dump({'schemaVersion': 1, 'method': 'regen-v1: Haiku draft, edge snap, Sonnet correction, edge snap', 'profiles': profiles}, open(root / 'profiles.json', 'w'))
print(stats)
