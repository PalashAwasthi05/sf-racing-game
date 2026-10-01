"""Final readings -> the profile file the generator takes (--regen-profiles). Readings measured on a camera whose
skyline registration passed are first moved onto the registered pose (remap.py)."""
import glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import specs, remap, colours, aligncorr
import frontage_crops_v15 as fc
# the skyline registrations moved readings by metres where the photographs already matched the walls (a side's windows
# came out 2.5 m left and 1.8 m low): off until a registration is checked against the wall it is used for
USE_REGISTRATION = False
root = pathlib.Path('regen/v1'); byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}
reg = {}
ALIGN = json.load(open(root / 'alignment.json')) if (root / 'alignment.json').exists() else {}
for f in glob.glob(str(root / 'registration*.json')): reg.update(json.load(open(f)))
world, frame = fc.load_world(); cams = fc.load_cameras(frame); bz = {b['id']: b['center'][2] for b in world['buildings']}
profiles, stats = [], {'profiles': 0, 'registered': 0, 'skipped': 0}
SRC = sys.argv[1] if len(sys.argv) > 1 else 'final'
for f in sorted((root / SRC).glob('*.json')):
    r = json.load(open(f)); p = byid[f.stem]
    if r.get('thisBuilding') is False or not r.get('materials'): stats['skipped'] += 1; continue
    r = specs.clean(r, p); img = p['views'][r['use']]['image']
    r = colours.measure(r, p, root / 'sides/img')
    r, c = aligncorr.correct(r, p, ALIGN.get(f.stem)); stats['aligned'] = stats.get('aligned', 0) + bool(c)
    if USE_REGISTRATION and remap.good(reg.get(img)):
        r = remap.remap_reading(r, remap.Map(p, cams[img], reg[img], bz[p['building']])); stats['registered'] += 1
    profiles.append(specs.to_profile(r, p)); stats['profiles'] += 1
json.dump({'schemaVersion': 1, 'method': 'regen-v1: Haiku draft, edge snap, Sonnet correction, edge snap', 'profiles': profiles}, open(root / 'profiles.json', 'w'))
print(stats)
