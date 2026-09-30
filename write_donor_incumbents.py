"""Write pipeline/donor-incumbents.json from a published candidate: for every building no photograph covers, the area
donor its walls were drawn from. pipeline/facades.donor_for keeps these donors in later builds while they still qualify,
so a change elsewhere in the reading pool (a re-read, a reading added or dropped) does not re-roll a building nobody
photographed. The donor is the facade records' donorId; a build from before that field existed gets area_donor recomputed
from the build's own profiles, kept only when it gives the material the building's walls were drawn in.
publish_full_candidate_v24.py runs this after every publish.
Usage: write_donor_incumbents.py [--candidate work/<dir>] (default: the live release)"""
import argparse, collections, datetime, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import facades                                   # noqa: E402
from pipeline.surface_evidence import resolve_material         # noqa: E402

READ = {'visual-fit', 'image-layout-fit', 'visual-grammar', 'envelope-graph'}


def read_wall(w):
    """A wall drawn from a reading of its own (hand, automatic or envelope), not from priors or a donor."""
    return w.get('edge') == 'grammar' or bool(w.get('grammarId')) or w.get('status') in READ


def frontage_read(w):
    """A wall drawn from a frontage reading of the building. An envelope design is not one: facades.compile still lends
    the building's other walls an area donor."""
    return w.get('edge') == 'grammar' or w.get('status') in ('visual-fit', 'image-layout-fit', 'visual-grammar')


def live_candidate():
    gen = json.load(open(ROOT / 'king-street/public/generation.json'))
    p = pathlib.Path(gen['candidate'])   # written relative to the workspace root
    return (p if p.is_absolute() else ROOT.parent / p).resolve().parent.parent


def evidence(cand):
    e = json.load(open(pathlib.Path(cand) / 'king-street-evidence.json'))
    return e['facadeWalls'], {b['id']: b for b in e['world']['buildings']}, e.get('profiles', [])


def donors_of(walls, buildings, profiles):
    """{building id: (donor profile, material)} for buildings drawn from an area donor, and the ids skipped with a reason."""
    by = collections.defaultdict(list)
    for w in walls: by[w['buildingId']].append(w)
    out, skipped = {}, {}
    for bid, ws in sorted(by.items()):
        if any(frontage_read(w) for w in ws): continue         # photographed: its walls follow its own readings
        drawn = [w for w in ws if w.get('status') in ('inferred', 'blank') and w.get('material')]
        if not drawn: continue
        mats = collections.Counter(w['material'] for w in drawn)
        named = collections.Counter(w['donorId'] for w in drawn if w.get('donorId'))
        if named:
            did = named.most_common(1)[0][0]
            # reading ids repeat across facings: the one whose material the walls were drawn in
            cands = [p for p in profiles if p.get('id') == did]
            donor = next((p for p in cands if resolve_material(p) in mats), cands[0] if cands else None)
        else:
            b = buildings.get(bid)
            if not b or not b.get('height'): skipped[bid] = 'no height in the build'; continue
            donor = facades.area_donor(profiles, re.sub(r'_p\d+$', '', bid), float(b['height']))
        if donor is None: skipped[bid] = 'no donor (prior only)'; continue
        m = resolve_material(donor)
        if m not in mats: skipped[bid] = f'recomputed donor {donor["id"]} gives {m}, walls are {mats.most_common(1)[0][0]}'; continue
        out[bid] = (donor, m)
    return out, skipped


GRID = ('floorPitchM', 'bayPitchM', 'openingRatio', 'openingHeightRatio', 'shopfront', 'ribbon')


def grid_of(donor):
    """The storey and bay grid lend() takes from a donor (facades.measured_grid, else its reading)."""
    r = facades.measured_grid(donor) or donor.get('reading') or {}
    return {k: r[k] for k in GRID if k in r}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path); ap.add_argument('--out', type=pathlib.Path, default=ROOT / 'pipeline/donor-incumbents.json')
    a = ap.parse_args(); cand = (a.candidate or live_candidate()).resolve()
    walls, buildings, profiles = evidence(cand)
    donors, skipped = donors_of(walls, buildings, profiles)
    rel = {}
    if (cand / 'release.json').exists():
        r = json.load(open(cand / 'release.json')); rel = {'contentHash': r.get('sha256', '')[:12]}
    doc = {
        'rule': 'Donors of the buildings no photograph covers, as published, with the material and grid each lent. pipeline/facades.donor_for keeps a building\'s donor and that material and grid while the donor is still an automatic reading with a measured grid, of the same facing, and within a factor of 1.6 of the building\'s height; otherwise area_donor chooses afresh. Set "keep" to false only for a rule change meant to choose donors afresh, and read review_inferred_changes.py\'s list for it. Written by write_donor_incumbents.py after each publish.',
        'keep': True,
        'release': {'candidate': str(cand.relative_to(ROOT.parent)) if cand.is_relative_to(ROOT.parent) else str(cand), **rel, 'written': datetime.date.today().isoformat()},
        'donors': {bid: {'donorId': d['id'], 'normal': d.get('normal'), 'heightRefM': d.get('heightRefM'), 'material': m, 'grid': grid_of(d)} for bid, (d, m) in donors.items()},
        'skipped': skipped,
    }
    a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(donors)} buildings keep their donor ({a.out}); {len(skipped)} skipped')
    for bid, why in list(skipped.items())[:12]: print('   skipped', bid, why)


if __name__ == '__main__':
    main()
