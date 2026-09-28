"""Frontage reading v34: v33's photographs that agree, with four faults of the readings fixed.

1. Readings of the wrong building. Another building (or a lower or taller part of this one) standing between camera
   and wall was read as the wall. Each photograph's rectified wall now carries a mask of what LiDAR building mass hides
   (frontage_evidence_v34.hidden_mask): hidden pixels count as occluded, the wall colour and the storey grid are read
   from the rest (auto_read_frontages_v24.read's exclusion mask), and detected windows, signs and glazing standing in
   the hidden area are dropped; columns the LiDAR mass behind the wall does not back to within 3 m of its top are not
   this building's face and count as hidden too (where the building's own mass stands further back, the rows above the
   lower mass in front are its set-back face: read for colour only, and such a photograph is read when 25 m2 of that
   face show). A photograph in which buildings hide more than 60 % of the wall is not read otherwise; a published
   reading whose own photograph had more than 35 % of the wall hidden is not kept on its standing (it is judged like a
   new reading), and it is replaced only by a reading that saw 40 % of its wall (a weaker witness does not overturn it;
   it stays, marked unsupported). Every automatic reading carries its building part (match.partId), so the drawing step applies it to
   that part (or to a part of similar height) instead of every part of the building.
2. Sun and shade. Readings agree when their colours can be one material under different light
   (pipeline/colour_agreement.same_material: hue and chroma kept, lightness free within 35 L*), where v33's dE 15 split
   one brick wall photographed in sun and in shade.
3. Odd windows. (Dropping windows the other photographs did not detect is implemented, OPENING_AGREEMENT, and off: it
   removed mostly real windows.) Every reading's upper-storey windows are laid out as storeys and bays where the
   detections support a
   pitch (frontage_evidence_v34.regular_layout; slivers dropped, windows snapped to the bays they imply, widths and
   storey heights unified), published readings included.
The v33 rules stand: up to five photographs per facing (another day or 10 m apart, score at least 40 % of the best);
a published reading (--incumbent) is kept as published unless three or more other photographs agree against it, or it
is near-black and two lit photographs agree; within an agreeing set the best-viewed photograph's reading is kept; new
facings that are near-black and uncorroborated, or disagree with the building's published or hand readings, are not
added, and cluttered ones (four or more windows without a storey pitch) give colour only; published readings of facings
no longer chosen are carried over; every photograph consulted is listed in sourceIds.
Usage: auto_read_frontages_v34.py --candidate work/<dir> --out work/pipeline/frontage-observations-auto.json
       --incumbent work/frontage-reading-incumbent-v032.json [--max-photos 5] [--max-walls 400]"""
import argparse, copy, json, math, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import auto_read_frontages_v27 as R27
import auto_read_frontages_v31 as V
import frontage_evidence_v34 as E
from auto_read_frontages_v24 import rectify, read, MIN_WALL_M
from auto_read_frontages_v33 import colour_of, dark, pick_photos, agreeing
from pipeline.colour_agreement import same_material

MAX_HIDDEN_READ, MAX_HIDDEN_INCUMBENT, MIN_SETBACK_M2 = .6, .35, 25.
# Off: dropping windows no other photograph detected and two clear ones did not removed 191 windows, mostly real (the
# judges' missing detail rose from 0.189 to 0.236 and 0.210 to 0.235): the detector misses too many windows for
# absence in a photograph to count against one. Kept for when detection recall allows it.
OPENING_AGREEMENT = False


def agree(a, b):
    ca, cb = colour_of(a), colour_of(b); return ca is not None and cb is not None and same_material(ca, cb)


def consensus(readings, keep_first=False):
    """v33's consensus with lighting-tolerant agreement. readings: [(candidate, profile)], the incumbent first when
    keep_first, then in view-score order. Returns (index kept, size of the agreeing group)."""
    ag = lambda i, j: agree(readings[i][1], readings[j][1]); best_viewed = min
    if keep_first:
        others = [i for i in range(1, len(readings)) if not dark(readings[i][1])]
        if dark(readings[0][1]):
            top = agreeing(others, ag); return (best_viewed(top), len(top)) if len(top) >= 2 else (0, 1)
        top = agreeing([0] + others, ag)
        if not top or 0 in top: return 0, max(1, len(top))
        if len(top) >= 3: return best_viewed(top), len(top)
        return 0, 1
    pool = [i for i, (_, p) in enumerate(readings) if not dark(p)] or list(range(len(readings)))
    top = agreeing(pool, ag) or [pool[0]]
    return best_viewed(top), len(top) if len(top) >= 2 else 1


def part_of(profile_id):
    return profile_id.split('-auto-')[0] if '-auto-' in profile_id else None


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
    surf = E.building_surface(world, frame); ground = E.ground_grid(world, frame)
    best = {}
    for w in walls:
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
    def incumbent_profile(w, bid):
        n = np.asarray(w['normal'], float); n /= np.linalg.norm(n); u = np.array([-n[1], n[0]]); mu = float(np.asarray(w['midpoint'], float) @ u)
        lo, hi = mu - w['lengthM'] / 2, mu + w['lengthM'] / 2; best_q, ov = None, 0.
        for q in incumbents.get(f"{bid}-auto-{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}", []):
            f = q.get('frameU') or [lo, hi]; o = min(hi, f[1]) - max(lo, f[0])
            if o > ov: best_q, ov = q, o
        return best_q
    seg = det = None; profiles = []; pano_cache = {}; used_incumbents = set()
    stats = {'facings': len(chosen), 'read': 0, 'blank': 0, 'materialOnly': 0, 'windows': 0, 'signs': 0, 'photosRead': 0, 'photosHidden': 0, 'keptBestScored': 0, 'keptOther': 0,
             'singlePhoto': 0, 'unagreed': 0, 'incumbentKept': 0, 'incumbentReplaced': 0, 'incumbentHidden': 0}
    for i, (score, w, cands) in enumerate(chosen):
        bid = w['buildingId']; z0 = buildings[bid]['center'][2]; H, L = float(w['heightM']), float(w['lengthM'])
        own = E.own_footprints(w, occ_fp); hidden = {}; sbk = {}; unbacked = E.unbacked_mask(w, z0, H, L, surf, own, sbk)
        def hidden_of(c):
            if c['image'] not in hidden:
                hm_ = E.hidden_mask(w, cams[c['image']], z0, H, L, surf, own, ground) | unbacked
                if sbk.get('share', 0.) > 0: hm_ |= E.setback_other(w, cams[c['image']], z0, H, L, surf, own, sbk['cols'], ground) & ~unbacked
                hidden[c['image']] = hm_
            return hidden[c['image']]
        inc_p = incumbent_profile(w, bid) if incumbents else None; inc = inc_p['camera']['image'] if inc_p else None
        if inc: used_incumbents.add((inc_p['id'], inc))
        inc_c = next((c for c in cands if c['image'] == inc), None)
        inc_hidden = float(hidden_of(inc_c).mean()) if inc_c else None
        if inc_hidden is not None and inc_hidden > MAX_HIDDEN_INCUMBENT: stats['incumbentHidden'] += 1; inc_standing = False
        else: inc_standing = inc_c is not None
        def usable_photo(c):
            share = float(hidden_of(c).mean())
            if share <= MAX_HIDDEN_READ or (c['image'] == inc and inc_standing): return True
            # a set-back face of this building, seen over the lower mass in front: read for its colour when 25 m2 of it show
            return sbk.get('visible', 0.) > .5 and (1. - share) * H * L >= MIN_SETBACK_M2
        usable = [c for c in cands[:16] if usable_photo(c)]
        stats['photosHidden'] += sum(1 for c in cands[:16] if c not in usable)
        readings = []; photos_found = []
        for c in pick_photos(usable, cams, a.max_photos, first=inc if inc_standing else None):
            cam = cams[c['image']]
            if c['image'] not in pano_cache:
                if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
                pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
                road = pano[int(pano.shape[0] * .74):int(pano.shape[0] * .9):4, ::4].reshape(-1, 3); road = road[(road.max(1) - road.min(1)) < 40]
                gain = np.clip((road.mean(0).mean() / np.maximum(road.mean(0), 1.)) if len(road) > 500 else np.ones(3), .85, 1.15)
                pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
            img, H_, L_ = rectify(w, cam, z0, pano_cache[c['image']]); hm = hidden_of(c)
            reading = read(img, H_, L_, exclude=hm)
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
            found = E.mask_found(found, hm, H_)
            p = V.profile(bid, buildings[bid], w, img, reading, c['image'], found)
            if sbk.get('visible', 0.) > .5 and not p.get('materialOnly'):   # what is seen is a set-back face: its colour, not a layout
                for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): p.pop(k_, None)
                p['materialOnly'] = True; p.setdefault('reading', {})['setbackFace'] = round(sbk['visible'], 2)
            p['camera'] = {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2), 'hiddenShare': round(float(hm.mean()), 2)}
            readings.append((c, p)); photos_found.append((found, img, reading, H_, L_)); stats['photosRead'] += 1
        if not readings: continue
        has_inc = inc_standing and readings[0][0]['image'] == inc
        k, n_agree = consensus(readings, keep_first=has_inc); p = readings[k][1]
        # Openings the other photographs contradict go: detected in no other photograph and absent from two or more that
        # see the place clearly. A new reading is rebuilt from its agreed detections.
        others = lambda j: [(photos_found[i][0][0], photos_found[i][0][2].astype(bool)) for i in range(len(photos_found)) if i != j]
        if OPENING_AGREEMENT and len(photos_found) >= 3 and not (has_inc and k == 0):
            f_k, img_k, reading_k, Hk, Lk = photos_found[k]; kept_ops, dropped_ops = E.agreed_openings(f_k[0], others(k), Hk)
            if dropped_ops:
                cam_k = readings[k][0]; p = V.profile(bid, buildings[bid], w, img_k, reading_k, cam_k['image'], (kept_ops,) + tuple(f_k[1:]))
                p['camera'] = readings[k][1]['camera']; p.setdefault('reading', {})['openingsUnagreed'] = dropped_ops; readings[k] = (cam_k, p)
                stats['openingsUnagreed'] = stats.get('openingsUnagreed', 0) + dropped_ops
        if has_inc and k == 0:
            p = copy.deepcopy(inc_p)
            if OPENING_AGREEMENT and len(photos_found) >= 3 and p.get('frameU') and p.get('heightRefM'):
                # the published layout's windows, in this wall's metres, against the other photographs' detections
                nn = np.asarray(w['normal'], float); nn /= np.linalg.norm(nn); uu = np.array([-nn[1], nn[0]]); left = float(np.asarray(w['midpoint'], float) @ uu) - L / 2
                f0_, f1_ = map(float, p['frameU']); href = float(p['heightRefM']); keep_o = []; gone = 0
                for o in p.get('openings', []):
                    x0, z0_, x1, z1 = o['rect']; box = (f0_ + x0 * (f1_ - f0_) - left, z0_ * href, f0_ + x1 * (f1_ - f0_) - left, z1 * href)
                    ok, against = E.corroborated(box, others(0), H) if o['style'].get('kind') == 'window' else (True, 0)
                    if ok or against < 2: keep_o.append(o)
                    else: gone += 1
                if gone:
                    p['openings'] = keep_o; p.setdefault('reading', {})['openingsUnagreed'] = gone; stats['openingsUnagreed'] = stats.get('openingsUnagreed', 0) + gone
        weak_replacement = bool(inc_p) and not has_inc and float((p.get('reading') or {}).get('visibleShare', 1.)) < .4
        if weak_replacement:
            # the published reading's photograph was mostly not this wall, but what is left to read is a weaker witness still:
            # the published reading stays (flagged unsupported) rather than churn the facade on a quarter-seen wall
            p = copy.deepcopy(inc_p); stats['publishedUnsupportedKept'] = stats.get('publishedUnsupportedKept', 0) + 1
        p['sourceIds'] = sorted(set(p.get('sourceIds', [])) | {'mapillary:' + c['image'] for c, _ in readings})
        nn = np.asarray(w['normal'], float); p['planeV'] = round(float(np.asarray(w['midpoint'], float) @ (nn / np.linalg.norm(nn))), 2)   # the wall's plane offset
        p.setdefault('reading', {})['consensus'] = {'photos': len(readings), 'kept': readings[k][0]['image'], 'keptRank': k, 'agreeing': n_agree, 'unagreed': len(readings) > 1 and n_agree < 2,
                                                    'incumbent': inc if (has_inc or weak_replacement) else None, 'incumbentHidden': round(inc_hidden, 2) if inc_hidden is not None and not inc_standing else None, 'unsupported': weak_replacement,
                                                    'replacesPublished': inc if inc and not has_inc and not weak_replacement else None,
                                                    'others': [{'image': c['image'], 'colorRgb': colour_of(q), 'storeys': (q.get('reading') or {}).get('storeys'), 'dark': dark(q)} for j, (c, q) in enumerate(readings) if j != k]}
        profiles.append(p)
        stats['read'] += 1; stats['singlePhoto'] += len(readings) == 1; stats['keptBestScored'] += len(readings) > 1 and k == 0; stats['keptOther'] += k > 0
        stats['unagreed'] += len(readings) > 1 and n_agree < 2; stats['incumbentKept'] += has_inc and k == 0; stats['incumbentReplaced'] += bool(inc) and not (has_inc and k == 0) and not weak_replacement
        print(i, bid, 'photos', len(readings), 'kept', k, 'agreeing', n_agree, colour_of(p), ('published photograph hidden' if inc_hidden is not None and inc_hidden > MAX_HIDDEN_INCUMBENT else 'published photograph not a candidate') if inc and not inc_standing else '', flush=True)
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
        c = (q.get('reading') or {}).get('consensus') or {}; new = not c.get('incumbent') and not c.get('replacesPublished')
        refs = published.get(q['match'].get('featureId'), [])
        if new and dark(q) and (c.get('agreeing') or 1) < 2:
            stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'near-black, uncorroborated', flush=True); continue
        r_ = q.get('reading') or {}
        if new and not q.get('materialOnly') and (r_.get('windows') or 0) >= 4 and not r_.get('floorPitchM'):
            for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): q.pop(k_, None)
            q['materialOnly'] = True; stats['newFacingsColourOnly'] = stats.get('newFacingsColourOnly', 0) + 1; print('colour only', q['id'], r_.get('windows'), 'windows, no storey pitch', flush=True)
        if not new or not refs or colour_of(q) is None or any(same_material(colour_of(q), r) for r in refs): kept_profiles.append(q)
        else: stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'disagrees with', refs, flush=True)
    profiles = kept_profiles
    carried = [q for qs in incumbents.values() for q in qs if (q['id'], q['camera']['image']) not in used_incumbents]
    for q in carried: q.setdefault('reading', {})['consensus'] = {'carriedOver': True}
    profiles += carried; stats['carriedOver'] = len(carried)
    layouts = {'profiles': 0, 'windowsSnapped': 0, 'sliversDropped': 0, 'withPitch': 0}
    for q in profiles:
        part = part_of(q['id'])
        if part: q['match']['partId'] = part                      # the building part this reading photographed
        lay = E.regular_layout(q)
        if lay:
            layouts['profiles'] += 1; layouts['windowsSnapped'] += lay['windowsSnapped']; layouts['sliversDropped'] += lay['sliversDropped']; layouts['withPitch'] += bool(lay['bayPitchM'])
        stats['materialOnly'] += bool(q.get('materialOnly')); stats['blank'] += bool((q.get('reading') or {}).get('blank'))
        stats['windows'] += len([o for o in q.get('openings', []) if o['style']['kind'] == 'window']); stats['signs'] += len(q.get('panels', []))
    stats['layouts'] = layouts
    a.out.write_text(json.dumps({'schemaVersion': 1, 'method': 'automatic rectified reading, agreeing photographs with building occlusion, part identity and regular layouts (auto_read_frontages_v34)',
                                 'profiles': profiles, 'stats': stats}, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else list(v) if isinstance(v, tuple) else str(v)))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
