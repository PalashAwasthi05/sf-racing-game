"""Frontage reading v35: v34, with the faces of blocks set back behind a lower part of their own building read.

v34 read one wall per building and facing, rectified from the ground, and skipped every facing a hand reading covered. A
block standing back on its podium (the podium's low wall hand-read on the same facing) was therefore never read: its
faces took the prior's punched windows. v35:
- keys walls by building, facing, plane (walls more than 2.5 m apart in depth, the grammar's frontage tolerance, are
  different frontages) and foot (zBaseM to the metre), so a podium wall and the block face above and behind it are read
  separately, and so is a block's face standing on a terrace within 2.5 m of the terrace's own wall below it;
- counts a facing as covered by a hand reading only on the plane that reading draws (within 2.5 m of the outline's
  outermost plane for that facing), and matches published readings (--incumbent) on their own plane (planeV);
- rectifies and masks a tier wall from its own foot (its evidence record's zBaseM, the roof it stands on), the camera
  still at eye height above the ground, and reads it without a ground storey (auto_read_frontages_v27.regularise
  ground=False); its readings carry zBaseM and planeV, and ids of set-back planes carry a -z<metres> suffix;
- reads a set-back tier wall's layout (the v34 colour-only rule is for outline walls whose building stands back from
  them, not for a wall that is itself the set-back face).
v34's notes follow.

Frontage reading v34: v33's photographs that agree, with four faults of the readings fixed.

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
Usage: auto_read_frontages_v35.py --candidate work/<dir> --out work/pipeline/frontage-observations-auto.json
       --incumbent work/frontage-reading-incumbent-v032.json [--max-photos 5] [--max-walls 400]"""
import argparse, copy, json, math, pathlib, re, sys
import numpy as np
from panorama_balance import road_gain
from recolour_readings_wb import recolour_profile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_crops_v15 as fc
import auto_read_frontages_v27 as R27
import auto_read_frontages_v31 as V
import frontage_evidence_v34 as E
import frontage_parts_v36 as FP
from auto_read_frontages_v24 import rectify, read, MIN_WALL_M
from auto_read_frontages_v33 import colour_of, dark, pick_photos, agreeing
from pipeline.colour_agreement import same_material, lab

MAX_HIDDEN_READ, MAX_HIDDEN_INCUMBENT, MIN_SETBACK_M2 = .6, .35, 25.
# The upper storeys of a part seen over a lower part of its own outline: a band of at least 10 m2 per photograph, read for
# its colour only, and kept only when two or more photographs agree (the project's standard for a photographed colour).
MIN_PART_BAND_M2, PART_BAND_AGREE = 10., 2


def band_family(rgb):
    """A dark, cool, unsaturated band above a lower part is glazing and metal cladding (the colour a curtain wall reads
    in daylight, HLS lightness up to .42 with the sky reflected), not grey render; anything else keeps the reader's family."""
    import colorsys
    h, l, s_ = colorsys.rgb_to_hls(*[float(x) / 255. for x in rgb]); hd = h * 360.
    if l < .42 and (s_ < .12 or 60. <= hd <= 260.): return ('metal-panel', 'dark')   # glass reflecting sky reads lighter than paint
    return None
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
        # Shade, haze and exposure take colour out of a wall, they never put it in: photographs that agree only by all
        # reading near-grey do not overturn a published reading that saw its wall's colour. Balanced against the road
        # (panorama_balance), a sunlit beige wall's shaded photographs read grey and outvoted it with a tree's shadow.
        chroma = lambda k: float(np.hypot(*lab(colour_of(readings[k][1]))[1:])) if colour_of(readings[k][1]) is not None else 0.
        if len(top) >= 3 and chroma(0) >= 12. and max(chroma(k) for k in top) <= chroma(0) / 1.5: return 0, 1
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
    ap.add_argument('--cache', type=pathlib.Path, default=pathlib.Path('work/frontage-reading-cache-v27'))
    # A section: read only these source buildings and merge the result into the existing --out file (other buildings'
    # readings are kept as they are). The rules are the same; this is for regenerating a reproduced section first.
    ap.add_argument('--only', nargs='*', default=None); a = ap.parse_args(); a.cache.mkdir(parents=True, exist_ok=True)
    only = set(a.only) if a.only else None
    world = json.load(open(a.candidate / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    buildings = {b['id']: b for b in world['buildings']}
    ev = json.load(open(a.candidate / 'public/facade-evidence.json'))
    walls = [w for w in ev['walls'] if w.get('lengthM', 0) >= MIN_WALL_M and w.get('heightM', 0) >= 3 and w.get('buildingId') in buildings and w.get('status') != 'envelope-graph'
             and not buildings[w['buildingId']].get('landmark') and (only is None or re.sub(r'_p\d+$', '', w['buildingId']) in only)]
    occ_fp = fc.Occluders(fc.load_footprints(frame)); trees = fc.load_trees(frame); cams = fc.load_cameras(frame)
    surf = E.building_surface(world, frame); ground = E.ground_grid(world, frame)
    # a split outline's parts are each other's occluders (frontage_parts_v36): the outline's own footprints are not one mass
    glb = a.candidate / 'materials/candidate.glb'; parts = FP.part_outlines(str(glb if glb.exists() else a.candidate / 'final/candidate.glb'))
    def plane_of(w):
        nn = np.asarray(w['normal'], float); return float(np.asarray(w['midpoint'], float) @ (nn / np.linalg.norm(nn)))
    facing_of = lambda w: int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])) / 10))
    # planes of one building and facing, clustered with a 2.5 m gap: each cluster is a frontage of its own
    clusters = {}
    for w in walls: clusters.setdefault((w['buildingId'], facing_of(w)), []).append(plane_of(w))
    plane_key = {}
    for k, vs in clusters.items():
        vs = sorted(vs, reverse=True); ci = 0
        for j, v in enumerate(vs):
            if j and vs[j - 1] - v > 2.5: ci += 1
            plane_key[(k, round(v, 3))] = ci
    best = {}
    members = {}
    for w in walls:
        cands = fc.wall_candidates(w, cams, occ_fp, trees, fc.MAX_DISTANCE_M)
        if not cands: continue
        # one reading per facing, plane and foot: a tier's face standing on the roof in front of it is rectified from that
        # foot and drawn only on walls standing there (zBaseM), so the block's face above a terrace is a frontage of its
        # own even where it stands within 2.5 m of the terrace's wall below
        k0 = (w['buildingId'], facing_of(w)); key = k0 + (plane_key[(k0, round(plane_of(w), 3))], int(round(float(w.get('zBaseM') or 0.)))); s = cands[0]['score'] * w['lengthM']
        if key not in best or s > best[key][0]: best[key] = (s, w, cands)
        members.setdefault(key, []).append(w); w['_frontage'] = key
    hand = []
    for f in sorted((pathlib.Path(__file__).resolve().parent / 'pipeline').glob('frontage-observations*.json')) + [pathlib.Path(__file__).resolve().parent / 'pipeline/facade-profiles.json']:
        if f.resolve() == a.out.resolve(): continue
        hand += json.load(open(f)).get('profiles', [])
    fps = fc.load_footprints(frame)
    def covered(w):
        # a hand reading draws the outline's outermost plane of its facing (within the grammar's 2.5 m): a wall standing
        # further back, the face of a block behind its podium, is not covered by it
        bid, normal = w['buildingId'], np.asarray(w['normal'], float); fid = re.sub(r'_p\d+$', '', bid)
        hands = [p for p in hand if p['match'].get('featureId') == fid and np.dot(normal, p['normal']) > .95]
        if not hands: return False
        g = fps.get(fid)
        if g is None: return True
        nn = normal / np.linalg.norm(normal); outer = max(float((np.asarray(q.exterior.coords) @ nn).max()) for q in (g.geoms if hasattr(g, 'geoms') else [g]))
        return plane_of(w) >= outer - 2.5
    chosen = sorted([v for v in best.values() if not covered(v[1])], key=lambda x: -x[0])[:a.max_walls]
    incumbents = {}
    if a.incumbent:
        for q in json.load(open(a.incumbent))['profiles']:
            if (q.get('camera') or {}).get('image'): incumbents.setdefault(q['id'], []).append(q)
    def incumbent_profile(w, bid):
        # the published reading of this frontage: matched against every wall of the frontage (facing, plane, foot), not
        # only the one photographed this run (a 3.5 m piece 5 m off the published plane lost a 15-storey reading)
        deg = int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0])))) % 360; zb_ = float(w.get('zBaseM') or 0.)
        pool = incumbents.get(f"{bid}-auto-{deg}" + (f"-z{int(round(zb_))}" if zb_ >= .5 else ''), []); best_q, ov = None, 0.
        for q in pool:
            qn = np.asarray(q['normal'], float)[:2]; qn = qn / np.linalg.norm(qn)       # planeV lies along the reading's own normal
            qu = np.array([-qn[1], qn[0]]); o = 0.
            for m in members.get(w.get('_frontage'), [w]):
                mid = np.asarray(m['midpoint'], float)
                if q.get('planeV') is not None and abs(float(q['planeV']) - float(mid @ qn)) > 2.5: continue   # another plane's reading
                mu = float(mid @ qu); lo, hi = mu - m['lengthM'] / 2, mu + m['lengthM'] / 2
                f = q.get('frameU') or [lo, hi]; o += max(0., min(hi, f[1]) - max(lo, f[0]))
            if o > ov: best_q, ov = q, o
        return best_q
    seg = det = None; profiles = []; pano_cache = {}; used_incumbents = set(); refreshed = {}
    stats = {'facings': len(chosen), 'read': 0, 'blank': 0, 'materialOnly': 0, 'windows': 0, 'signs': 0, 'photosRead': 0, 'photosHidden': 0, 'keptBestScored': 0, 'keptOther': 0,
             'singlePhoto': 0, 'unagreed': 0, 'incumbentKept': 0, 'incumbentReplaced': 0, 'incumbentHidden': 0}
    for i, (score, w, cands) in enumerate(chosen):
        bid = w['buildingId']; z0 = buildings[bid]['center'][2]; H, L = float(w['heightM']), float(w['lengthM'])
        zb = float(w.get('zBaseM') or 0.); zw = z0 + zb          # a tier wall's foot: the roof it stands on
        own = E.own_footprints(w, occ_fp); hidden = {}; sbk = {}; unbacked = E.unbacked_mask(w, zw, H, L, surf, own, sbk)
        # What hides the wall in front is judged against the wall's own part: another part of its split outline standing
        # between camera and wall (a low building before a taller one's face) hides it up to that part's full top. The
        # backing and set-back tests keep the whole outline (they ask what stands behind the wall).
        own_hid = FP.own_mass(w, occ_fp, parts, E.own_footprints); surf_hid = FP.raised_surface(surf, w, parts) if own_hid is not own else surf
        by_part = {}
        def hidden_of(c):
            if c['image'] not in hidden:
                hm_ = E.hidden_mask(w, cams[c['image']], zw, H, L, surf_hid, own_hid, ground) | unbacked
                if own_hid is not own: by_part[c['image']] = float((hm_ & ~(E.hidden_mask(w, cams[c['image']], zw, H, L, surf, own, ground) | unbacked)).mean())
                if sbk.get('share', 0.) > 0: hm_ |= E.setback_other(w, cams[c['image']], zw, H, L, surf, own, sbk['cols'], ground) & ~unbacked
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
            if sbk.get('visible', 0.) > .5 and (1. - share) * H * L >= MIN_SETBACK_M2: return True
            # the upper storeys of a taller part seen over a lower part of its own outline: read for colour when 10 m2 show
            return by_part.get(c['image'], 0.) >= .3 and (1. - share) * H * L >= MIN_PART_BAND_M2
        usable = [c for c in cands[:16] if usable_photo(c)]
        stats['photosHidden'] += sum(1 for c in cands[:16] if c not in usable)
        readings = []; photos_found = []
        for c in pick_photos(usable, cams, a.max_photos, first=inc if inc_standing else None):
            cam = cams[c['image']]
            if c['image'] not in pano_cache:
                if len(pano_cache) >= 3: pano_cache.pop(next(iter(pano_cache)))
                pano = np.asarray(fc.load_pano(cam['path']), dtype=np.float32)
                gain = road_gain(pano)   # the road below the horizon, not the capture vehicle (panorama_balance)
                pano_cache[c['image']] = np.clip(pano * gain, 0, 255).astype(np.uint8)
            img, H_, L_ = rectify(w, cam, z0, pano_cache[c['image']], base=zb); hm = hidden_of(c)
            reading = read(img, H_, L_, exclude=hm)
            ck = a.cache / (f"{re.sub(r'[^A-Za-z0-9_]', '', bid)}_{int(round(math.degrees(math.atan2(w['normal'][1], w['normal'][0]))))%360}_{c['image']}"
                            + (f"_z{int(round(zb))}h{int(round(H))}" if zb >= .5 else ''))   # a tier wall's image depends on its foot and height
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
            if zb >= .5:
                p['zBaseM'] = round(zb, 2); p['id'] += f"-z{int(round(zb))}"
            if zb < .5 and by_part.get(c['image'], 0.) >= .3 and float(hm.mean()) > MAX_HIDDEN_READ:   # a band over its own lower part: its colour
                for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): p.pop(k_, None)
                p['materialOnly'] = True; p.setdefault('reading', {})['aboveOwnPart'] = round(1. - float(hm.mean()), 2)
                ev_ = p.get('materialEvidence')
                if ev_ and ev_.get('colorRgb') and band_family(ev_['colorRgb']):
                    ev_['family'], ev_['finish'] = band_family(ev_['colorRgb']); p['reading']['curtainBand'] = True   # glazing and metal: drawn as a curtain wall
            if zb < .5 and sbk.get('visible', 0.) > .5 and not p.get('materialOnly'):   # what is seen is a set-back face: its colour, not a layout
                for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): p.pop(k_, None)
                p['materialOnly'] = True; p.setdefault('reading', {})['setbackFace'] = round(sbk['visible'], 2)
            p['camera'] = {'image': c['image'], 'distanceM': round(c['distanceM'], 1), 'incidence': round(c['incidence'], 2), 'hiddenShare': round(float(hm.mean()), 2)}
            if by_part.get(c['image']): p['camera']['hiddenByOwnParts'] = round(by_part[c['image']], 2)
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
            p, _ = recolour_profile(p, [0])     # panels, bands and zones by the photograph's gain ratio (recolour_readings_wb)
            # the published layout stays; its colour is read again from the same photograph, now white-balanced against
            # the road (panorama_balance): the published colour carried the capture vehicle's magenta cast
            fresh = readings[0][1].get('materialEvidence')
            if fresh and fresh.get('colorRgb') and p.get('materialEvidence'):
                p['materialEvidence'] = {**p['materialEvidence'], 'colorRgb': fresh['colorRgb'], 'family': fresh.get('family', p['materialEvidence'].get('family')), 'finish': fresh.get('finish', p['materialEvidence'].get('finish'))}
                refreshed[inc_p['id']] = fresh['colorRgb']
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
    # A new reading must agree with the published and hand readings of its own part: an automatic reading names its part
    # (partId or its id), a hand reading belongs to the parts whose height is within 30 % of its frame (as the drawing step
    # decides). Compared with the whole outline, a tall block's glass was withheld for disagreeing with the low brick
    # building in front of it.
    published = {}
    def part_ref(q):
        return q['match'].get('partId') or part_of(q['id'])
    for q in [q for qs in incumbents.values() for q in qs] + hand:
        rgb = refreshed.get(q['id']) or material_rgb(q)     # a published reading read again this run speaks in its new colour
        if rgb: published.setdefault(q['match'].get('featureId'), []).append((part_ref(q), q.get('heightRefM'), rgb, q))
    def frame_on(q, part):
        # a reading's frame (frameU along its plane planeV) lies on this part: at least half of it within 1 m of the
        # part's outline (as facades.frame_on_part asks before drawing it there); a reading without a frame counts
        g = parts.get(part)
        if g is None or not q.get('frameU') or q.get('planeV') is None or q.get('normal') is None: return True
        from shapely.geometry import LineString
        nn = np.asarray(q['normal'], float)[:2]; nn /= np.linalg.norm(nn); uu = np.array([-nn[1], nn[0]]); f0, f1 = map(float, q['frameU'])
        seg = LineString([nn * float(q['planeV']) + uu * f0, nn * float(q['planeV']) + uu * f1])
        return seg.length <= 0 or seg.intersection(g.buffer(1.)).length >= .5 * seg.length
    def refs_of(q):
        part = part_ref(q); ph = float((buildings.get(part) or {}).get('height') or 0.)
        out = []
        for rp, href, rgb, r in published.get(q['match'].get('featureId'), []):
            if rp is not None: ok = rp == part and frame_on(r, part)
            else: ok = not ph or not href or abs(float(href) - ph) <= .3 * max(float(href), ph)
            if ok: out.append(rgb)
        return out
    kept_profiles = []
    for q in profiles:
        c = (q.get('reading') or {}).get('consensus') or {}; new = not c.get('incumbent') and not c.get('replacesPublished')
        refs = refs_of(q)
        if new and (q.get('reading') or {}).get('aboveOwnPart') is not None and (c.get('agreeing') or 1) < PART_BAND_AGREE:
            stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'band over its own lower part, agreed by', c.get('agreeing'), flush=True); continue
        if new and dark(q) and (c.get('agreeing') or 1) < 2:
            stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'near-black, uncorroborated', flush=True); continue
        r_ = q.get('reading') or {}
        if new and not q.get('materialOnly') and (r_.get('windows') or 0) >= 4 and not r_.get('floorPitchM'):
            for k_ in ('openings', 'groups', 'panels', 'bands', 'piers'): q.pop(k_, None)
            q['materialOnly'] = True; stats['newFacingsColourOnly'] = stats.get('newFacingsColourOnly', 0) + 1; print('colour only', q['id'], r_.get('windows'), 'windows, no storey pitch', flush=True)
        if not new or not refs or colour_of(q) is None or any(same_material(colour_of(q), r) for r in refs): kept_profiles.append(q)
        else: stats['newFacingsWithheld'] = stats.get('newFacingsWithheld', 0) + 1; print('withheld', q['id'], colour_of(q), 'disagrees with', refs, flush=True)
    profiles = kept_profiles
    carried = [q for qs in incumbents.values() for q in qs if (q['id'], q['camera']['image']) not in used_incumbents
               and (only is None or q['match'].get('featureId') in only)]
    # a published reading of a facing not chosen this run is not read again: its colours move to the road white balance by
    # its photograph's gain ratio (recolour_readings_wb); kept verbatim, 28 of them still carried the capture vehicle's cast
    carried = [recolour_profile(q, [0])[0] for q in carried]
    for q in carried: q.setdefault('reading', {})['consensus'] = {**(q['reading'].get('consensus') or {}), 'carriedOver': True}   # keeps 'unsupported'
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
    if only is not None:
        # a section run: the other buildings' readings stay as they were in the output file
        prior = json.load(open(a.out))['profiles'] if a.out.exists() else []
        profiles = [q for q in prior if q['match'].get('featureId') not in only] + profiles; stats['section'] = sorted(only)
    a.out.write_text(json.dumps({'schemaVersion': 1, 'method': 'automatic rectified reading, agreeing photographs with building occlusion, part identity, regular layouts and set-back faces read on their own plane (auto_read_frontages_v35)',
                                 'profiles': profiles, 'stats': stats}, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else list(v) if isinstance(v, tuple) else str(v)))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
