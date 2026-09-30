"""What changed on the walls no photograph covers, candidate against live: for every building, each facing whose walls
are drawn from priors, a donor or the building's other readings in both builds, its material (the largest by wall area)
and archetype before and after, and why it changed: the facing was rebuilt (geometry), the building's own readings
changed (ids or materials), its height changed, its published donor is no longer in the pool, or donors were chosen
afresh on purpose ("keep": false in pipeline/donor-incumbents.json). A change with none of
these causes is a rule change nobody asked these walls for (the user's "regressed again": a wall no photograph checks
flips with every fallback change): the exit status is 2, and publish_full_candidate_v24.py refuses it without
--accept-inferred-changes. Photographed walls are left to the photo judge.
Writes <candidate>/inferred-changes.json.
Usage: review_inferred_changes.py --candidate work/<dir> [--live work/<dir>]"""
import argparse, collections, json, math, os, pathlib, re, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import write_donor_incumbents as W               # noqa: E402


def facings(walls, bid):
    """{facing (degrees, 20-degree bins): (largest material by area, archetype, area m2)} of a building's walls."""
    acc = collections.defaultdict(lambda: collections.Counter()); arch = {}
    for w in walls:
        if w['buildingId'] != bid or not w.get('material'): continue
        f = int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 20.)) * 20 % 360
        acc[f][w['material']] += w['lengthM'] * w['heightM']; arch.setdefault(f, collections.Counter())[w.get('archetype')] += w['lengthM'] * w['heightM']
    return {f: (c.most_common(1)[0][0], arch[f].most_common(1)[0][0], round(sum(c.values()), 1)) for f, c in acc.items()}


def owned(profiles, bid):
    """The readings filed under a building (its part, feature or map way), whether or not they drew a wall of their own."""
    src = re.sub(r'_p\d+$', '', bid)
    keys = {bid, src, src.replace('osm_', '')}
    return sorted(p['id'] for p in profiles if any(str(v) in keys for v in (p.get('match') or {}).values()))


def qualifies(donor, profiles):
    """The published donor is still an automatic reading with a grid, of the same facing (as facades.donor_for asks)."""
    for p in profiles:
        r = p.get('reading') or {}
        if p.get('id') != donor.get('id') or not str(p.get('status', '')).startswith('image-measured-auto') or not (r.get('floorPitchM') and r.get('bayPitchM') and p.get('heightRefM')): continue
        if donor.get('normal') is not None and p.get('normal') is not None and float(np.dot(p['normal'][:2], donor['normal'][:2])) < .98: continue
        return True
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--live', type=pathlib.Path)
    a = ap.parse_args(); cand = a.candidate.resolve(); live = (a.live or W.live_candidate()).resolve()
    lw, lb, lp = W.evidence(live); cw, cb, cp = W.evidence(cand)
    ld, _ = W.donors_of(lw, lb, lp); cd, _ = W.donors_of(cw, cb, cp)
    keep = True
    inc = pathlib.Path(os.environ.get('DONOR_INCUMBENTS') or ROOT / 'pipeline/donor-incumbents.json')
    incumbents = {}
    if inc.exists(): doc = json.load(open(inc)); keep = doc.get('keep', True); incumbents = doc.get('donors', {})
    from pipeline.surface_evidence import resolve_material
    def readings(profiles, bid):
        """(id, material) of the readings filed under a building: what its unseen walls may legitimately follow."""
        ids = set(owned(profiles, bid))
        return sorted((p['id'], resolve_material(p)) for p in profiles if p['id'] in ids)
    unseen = lambda walls: [w for w in walls if not W.read_wall(w)]
    lu, cu = unseen(lw), unseen(cw)
    ids = sorted({w['buildingId'] for w in lw} & {w['buildingId'] for w in cw})
    out = []
    for bid in ids:
        fl, fc = facings(lu, bid), facings(cu, bid)
        changed = {f: (fl[f], fc[f]) for f in set(fl) & set(fc) if fl[f][:2] != fc[f][:2]}
        rebuilt = sorted(set(fl) ^ set(fc))
        if not changed and not rebuilt: continue
        dl, dc = ld.get(bid), cd.get(bid)
        donor_l, donor_c = (dl[0]['id'] if dl else None), (dc[0]['id'] if dc else None)
        own_l, own_c = readings(lp, bid), readings(cp, bid)
        if not changed: cause, explained = 'facing rebuilt (geometry)', True
        elif own_l != own_c: cause, explained = 'its own readings changed', True
        elif dl is None and dc is None:
            hl, hc = float((lb.get(bid) or {}).get('height') or 0.), float((cb.get(bid) or {}).get('height') or 0.)
            cause, explained = (('building height changed', True) if hl and abs(hc / hl - 1) > .1 else
                                ('unseen walls changed, the building\'s readings did not: rule change', False))
        elif donor_l != donor_c:
            if not keep: cause, explained = 'donors chosen afresh ("keep": false)', True
            elif dl is None: cause, explained = 'now drawn from a donor (had none)', False
            else:
                hl, hc = float((lb.get(bid) or {}).get('height') or 0.), float((cb.get(bid) or {}).get('height') or 0.)
                if not qualifies(dl[0], cp): cause, explained = 'published donor no longer in the pool', True
                elif hl and abs(math.log(max(hc, 1.) / max(hl, 1.))) > math.log(1.3): cause, explained = 'building height changed (donor re-chosen)', True
                else: cause, explained = 'donor re-chosen without cause', False
        else:
            # with its published material and grid carried (pipeline/facades.donor_for), a kept donor cannot change the
            # walls: a change here means the carry was not applied
            cause, explained = (('the donor is kept but its published material or grid was not', False) if keep and bid in incumbents
                                else ("the donor's own reading changed", True))
        out.append({'building': bid, 'name': (cb.get(bid) or {}).get('name'), 'cause': cause, 'explained': explained, 'donor': [donor_l, donor_c],
                    'facings': {str(f): {'live': list(v[0]), 'candidate': list(v[1])} for f, v in sorted(changed.items())}, 'rebuiltFacings': rebuilt})
    summary = collections.Counter(e['cause'] for e in out)
    unexplained = [e['building'] for e in out if not e['explained']]
    res = {'candidate': str(cand), 'live': str(live), 'buildings': len(out), 'byCause': dict(summary), 'unexplained': unexplained, 'changes': out}
    (cand / 'inferred-changes.json').write_text(json.dumps(res, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(out)} unphotographed buildings changed ({cand / "inferred-changes.json"}):')
    for k, n in summary.most_common(): print(f'   {n:3d}  {k}')
    for e in out:
        if not e['explained']:
            for f, v in e['facings'].items(): print(f"   UNEXPLAINED {e['building']} ({e['name']}) facing {f}: {v['live'][0]} -> {v['candidate'][0]}  donor {e['donor'][0]} -> {e['donor'][1]}")
    sys.exit(2 if unexplained else 0)


if __name__ == '__main__':
    main()
