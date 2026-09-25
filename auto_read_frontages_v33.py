"""Frontage reading v33: a facing's reading is the one its photographs agree on, not whatever its best-scored
photograph shows. v31 read each facing from the single camera with the best view score (distance, incidence,
visible share, field of view). Once many more panoramas were available that choice kept switching to photographs
that scored a little better and read much worse: a wall read grey from 36 m where the 11 m photograph shows brick,
a wall read near-black in shade, storeys lost behind trees, a neighbour read through a pose error. A single
photograph cannot say which of those it is; several can.

For each facing, up to five photographs are read (v31's reading and components, cached detections): candidates in
view-score order, the first always, each further one only if it was taken on another day or at least 10 m from the
photographs already chosen (frames of one drive share its light, occluders and pose error), and only if its score is
at least 40 % of the best. Readings of a near-black wall (L* < 22: shade, a dark gap) are set aside while any other
reading exists. Of the rest, the largest set whose wall colours all agree pairwise within dE 15 (lightness at half
weight: the same wall under different light stays within it, a neighbour or a shadow does not) is the wall; a tie
goes to the set holding the better-scored photographs. Within it the best-viewed photograph's reading is kept: the
photographs agree on what the wall is, and the best view reads its layout best (keeping the reading with most windows
rewarded clutter: palm trunks and sign frames read as windows). When no two photographs agree the best-scored reading is kept and the
profile is marked unagreed.
An established reading (--incumbent: the published readings) is evidence too: its photograph is read first, and the
published profile is kept as it was published (its frame, layout and components) unless three or more other photographs agree against it (more than agree with it), or it is near-black
and two or more other photographs agree on a lit colour; a pair of photographs or a lone lit one does not overturn it.
A new facing whose reading is near-black and not corroborated by a second photograph is not added (one shaded
photograph is not evidence of a black wall). A new facing with four or more detected windows whose storeys do not
stack at a measured pitch (detections that are clutter: palm trunks, sign frames, reflections) gives its wall colour
only; the wall keeps the regular openings of its archetype instead of a scatter of odd windows. A new facing on a building that already has published or hand readings is
added only if its wall colour agrees with
one of them (dE < 15): the building's unseen walls continue its readings' material when they agree, and one new
reading that disagrees would drop them all to the prior (a dark purple-grey building turned white that way).
Published readings of facings no longer chosen (the building's parts were renamed, or the wall is now another
facing's) are carried over unchanged: the drawing step gives each wall the reading covering most of it. New
photographs then confirm or outvote a reading instead of replacing it because they score a little better (the
photograph-by-photograph switch of v31 turned walls that were right grey, dark or windowless). Every photograph consulted is listed in the profile's sourceIds (and so in the facade
evidence), so a judge can hold them all out.
Usage: auto_read_frontages_v33.py --candidate work/<dir> --out work/pipeline/frontage-observations-auto.json
       [--incumbent work/frontage-reading-incumbent-v032.json] [--max-photos 5] [--max-walls 400]"""
import argparse, copy, json, math, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import auto_read_frontages_v27 as R27
import auto_read_frontages_v31 as V
from auto_read_frontages_v24 import rectify, read, MIN_WALL_M
from judge_facades_v27 import to_lab

DARK_L, SEPARATION_M, MIN_SCORE_SHARE, COLOUR_AGREE = 22., 10., .4, 15.


def lab(rgb): return to_lab(np.asarray(rgb, float).reshape(1, 1, 3))[0, 0]


def colour_of(p):
    return (p.get('materialEvidence') or {}).get('colorRgb') or (p.get('reading') or {}).get('wallRgb')


def colour_de(a, b):
    ca, cb = colour_of(a), colour_of(b)
    if ca is None or cb is None: return 1e9
    la, lb = lab(ca), lab(cb); return math.sqrt(((la[0] - lb[0]) * .5) ** 2 + (la[1] - lb[1]) ** 2 + (la[2] - lb[2]) ** 2)


def dark(p):
    c = colour_of(p); return c is not None and lab(c)[0] < DARK_L


def pick_photos(cands, cams, k, first=None):
    """Up to k candidates: the incumbent's photograph (`first`) if it is still a candidate, then the best, then others
    taken on another day or at least SEPARATION_M from those chosen."""
    chosen = [c for c in cands if c['image'] == first][:1]
    for c in cands[:12]:
        if len(chosen) == k: break
        if c in chosen: continue
        if c['score'] < MIN_SCORE_SHARE * cands[0]['score']: break
        cam = cams[c['image']]
        if all(cams[o['image']]['captureDate'] != cam['captureDate'] or np.linalg.norm(cams[o['image']]['xy'] - cam['xy']) >= SEPARATION_M for o in chosen):
            chosen.append(c)
    return chosen


def agreeing(items, agree):
    """The largest set of items that all agree with one another (every pair), preferring the set holding the earliest
    (better-scored) items; [] when no two agree. Items are few (five photographs at most), so every set is tried."""
    from itertools import combinations
    for size in range(len(items), 1, -1):
        for g in combinations(items, size):              # lexicographic: sets holding earlier items come first
            if all(agree(i, j) for i, j in combinations(g, 2)): return list(g)
    return []


def consensus(readings, keep_first=False):
    """readings: [(candidate, profile)], the incumbent first when keep_first, then in view-score order. Returns (index
    kept, size of the agreeing group)."""
    agree = lambda i, j: colour_de(readings[i][1], readings[j][1]) < COLOUR_AGREE
    best_viewed = min                                   # readings are in view-score order (the incumbent first)
    if keep_first:
        others = [i for i in range(1, len(readings)) if not dark(readings[i][1])]
        if dark(readings[0][1]):                         # a near-black incumbent yields to two lit photographs that agree
            top = agreeing(others, agree)
            return (best_viewed(top), len(top)) if len(top) >= 2 else (0, 1)
        top = agreeing([0] + others, agree)              # ties go to sets holding the incumbent
        if not top or 0 in top: return 0, max(1, len(top))
        if len(top) >= 3: return best_viewed(top), len(top)
        return 0, 1
    pool = [i for i, (_, p) in enumerate(readings) if not dark(p)] or list(range(len(readings)))
    top = agreeing(pool, agree) or [pool[0]]
    return best_viewed(top), len(top) if len(top) >= 2 else 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--max-walls', type=int, default=400); ap.add_argument('--max-photos', type=int, default=5); ap.add_argument('--incumbent', type=pathlib.Path)
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('work/frontage-reading-cache-v27')); a = ap.parse_args(); a.cache.mkdir(parents=True, exist_ok=True)
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.candidate / 'public/facade-evidence.json'))
    walls = [w for w in ev['walls'] if w.get('lengthM', 0) >= MIN_WALL_M and w.get('heightM', 0) >= 3 and w.get('buildingId') in buildings and w.get('status') != 'envelope-graph'
             and not buildings[w['buildingId']].get('landmark')]
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    best = {}
    for w in walls:                                          # the facing set and each facing's wall, as v27 chooses them
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        key = (w['buildingId'], int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 10))); s = cands[0]['score'] * w['lengthM']
        if key not in best or s > best[key][0]: best[key] = (s, w, cands)
    hand = []
    for f in sorted((pathlib.Path(__file__).resolve().parent / 'pipeline').glob('frontage-observations*.json')) + [pathlib.Path(__file__).resolve().parent / 'pipeline/facade-profiles.json']:
        if f.resolve() == a.out.resolve(): continue
        hand += json.load(open(f)).get('profiles', [])
    def covered(bid, normal):
        fid = re.sub(r'_p\d+$', '', bid)
        return any(p['match'].get('featureId') == fid and np.dot(normal, p['normal']) > .95 for p in hand)
    chosen = sorted([v for v in best.values() if not covered(v[1]['buildingId'], v[1]['normal'])], key=lambda x: -x[0])[:a.max_walls]
    incumbents = {}
    if a.incumbent:
        for q in json.load(open(a.incumbent))['profiles']:
            if (q.get('camera') or {}).get('image'): incumbents.setdefault(q['id'], []).append(q)
    def incumbent_camera(w, bid):
        """The published reading's photograph for this wall: same profile id, most overlap along the frontage."""
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n); u = np.array([-n[1], n[0]]); mu = float(np.asarray(w['midpoint'], float) @ u)
        lo, hi = mu - w['lengthM'] / 2, mu + w['lengthM'] / 2; best, ov = None, 0.
        for q in incumbents.get(f"{bid}-auto-{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}", []):
            f = q.get('frameU') or [lo, hi]; o = min(hi, f[1]) - max(lo, f[0])
            if o > ov: best, ov = q, o
        return best
    seg = det = None; profiles = []; pano_cache = {}; used_incumbents = set()
    stats = {'facings': len(chosen), 'read': 0, 'blank': 0, 'materialOnly': 0, 'windows': 0, 'signs': 0, 'photosRead': 0, 'keptBestScored': 0, 'keptOther': 0, 'singlePhoto': 0, 'unagreed': 0, 'incumbentKept': 0, 'incumbentReplaced': 0}
    for i, (score, w, cands) in enumerate(chosen):
        bid = w['buildingId']; z0 = buildings[bid]['center'][2]; readings = []
        inc_p = incumbent_camera(w, bid) if incumbents else None; inc = inc_p['camera']['image'] if inc_p else None
        if inc: used_incumbents.add((inc_p['id'], inc))
        for c in pick_photos(cands, cams, a.max_photos, first=inc):
            cam = cams[c['image']]
            if c['image'] not in pano_cache:
                if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
                pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
                road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
                gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
                pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
            img, H, L = rectify(w, cam, z0, pano_cache[c['image']]); reading = read(img, H, L)
            ck = a.cache / f"{re.sub(r'[^A-Za-z0-9_]', '', bid)}_{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}_{c['image']}"
            if ck.with_suffix('.json').exists():
                j = json.loads(ck.with_suffix('.json').read_text()); occ = np.load(ck.with_suffix('.npz'))['occ']
                found = ([tuple(o) for o in j['ops']], [tuple(o) for o in j['signs']], occ, j['visible'], (img * 255).astype(np.uint8), [tuple(g) for g in j['glass']])
            else:
                if seg is None: seg, det = R27.Segmenter(), R27.Detector()
                found = R27.detect(img, seg, det)
                ck.with_suffix('.json').write_text(json.dumps({'ops': [[v if isinstance(v, str) else float(v) for v in o] for o in found[0]], 'signs': [[v if isinstance(v, str) else float(v) for v in o] for o in found[1]],
                                                              'visible': found[3], 'glass': [list(map(float, g)) for g in found[5]]}))
                np.savez_compressed(ck.with_suffix('.npz'), occ=found[2])
            p = V.profile(bid, buildings[bid], w, img, reading, c['image'], found)
            p['camera'] = {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2)}
            readings.append((c, p)); stats['photosRead'] += 1
        if not readings: continue
        has_inc = inc is not None and readings[0][0]['image'] == inc
        k, agree = consensus(readings, keep_first=has_inc); p = readings[k][1]
        if has_inc and k == 0: p = copy.deepcopy(inc_p)     # the published profile as published: frame, layout, components
        p['sourceIds'] = sorted(set(p.get('sourceIds', [])) | {'mapillary:' + c['image'] for c, _ in readings})
        p.setdefault('reading', {})['consensus'] = {'photos': len(readings), 'kept': readings[k][0]['image'], 'keptRank': k, 'agreeing': agree, 'unagreed': len(readings) > 1 and agree < 2, 'incumbent': inc if has_inc else None,
                                                    'others': [{'image': c['image'], 'colorRgb': colour_of(q), 'storeys': (q.get('reading') or {}).get('storeys'), 'dark': dark(q)} for j, (c, q) in enumerate(readings) if j != k]}
        profiles.append(p)
        stats['read'] += 1; stats['materialOnly'] += bool(p.get('materialOnly')); stats['blank'] += bool(p.get('reading', {}).get('blank'))
        stats['windows'] += len([o for o in p.get('openings', []) if o['style']['kind'] == 'window']); stats['signs'] += len(p.get('panels', []))
        stats['singlePhoto'] += len(readings) == 1; stats['keptBestScored'] += len(readings) > 1 and k == 0; stats['keptOther'] += k > 0; stats['unagreed'] += len(readings) > 1 and agree < 2; stats['incumbentKept'] += has_inc and k == 0; stats['incumbentReplaced'] += has_inc and k > 0
        print(i, bid, 'photos', len(readings), 'kept', k, 'agreeing', agree, colour_of(p), flush=True)
    # New facings of already-read buildings join only when they agree with a published or hand reading of the building.
    from pipeline.surface_evidence import resolve_material
    from pipeline import material_recipes
    def material_rgb(q):
        m = resolve_material(q) or ''; base, _, hx = m.partition('#')
        if len(hx) == 6: return [int(hx[i:i + 2], 16) for i in (0, 2, 4)]
        return (material_recipes.RECIPES.get(base) or {}).get('color')
    published = {}
    for q in [q for qs in incumbents.values() for q in qs] + hand:
        rgb = material_rgb(q)
        if rgb: published.setdefault(q['match'].get('featureId'), []).append(rgb)
    kept_profiles = []
    for q in profiles:
        c = (q.get('reading') or {}).get('consensus') or {}
        refs = published.get(q['match'].get('featureId'), [])
        if not c.get('incumbent') and not c.get('carriedOver') and dark(q) and (c.get('agreeing') or 1) < 2:
            stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'near-black, uncorroborated', flush=True); continue
        r_ = q.get('reading') or {}
        if not c.get('incumbent') and not c.get('carriedOver') and not q.get('materialOnly') and (r_.get('windows') or 0) >= 4 and not r_.get('floorPitchM'):
            for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): q.pop(k_, None)
            q['materialOnly'] = True; stats['newFacingsColourOnly'] = stats.get('newFacingsColourOnly', 0) + 1; print('colour only', q['id'], r_.get('windows'), 'windows, no storey pitch', flush=True)
        if c.get('incumbent') or not refs or colour_of(q) is None or any(colour_de(q, {'materialEvidence': {'colorRgb': r}}) < COLOUR_AGREE for r in refs):
            kept_profiles.append(q)
        else: stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'disagrees with', refs, flush=True)
    profiles = kept_profiles
    carried = [q for qs in incumbents.values() for q in qs if (q['id'], q['camera']['image']) not in used_incumbents]
    for q in carried: q.setdefault('reading', {})['consensus'] = {'carriedOver': True}
    profiles += carried; stats['carriedOver'] = len(carried)
    a.out.write_text(json.dumps({'schemaVersion': 1, 'method': 'automatic rectified reading, colour-agreeing photographs of up to %d (auto_read_frontages_v33)' % a.max_photos, 'profiles': profiles, 'stats': stats}, indent=1,
                                default=lambda v: v.item() if hasattr(v, 'item') else list(v) if isinstance(v, tuple) else str(v)))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
