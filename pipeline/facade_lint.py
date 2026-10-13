"""Coherence rules for a facade reading before the generator draws it: what a photograph reading can get wrong and the
generator must not draw. Each rule repairs or drops the parts that break it and says why; nothing location-specific.

  bands      a band (string course, sill or lintel course, timber beam) never runs through a window or door: it is cut
             around every opening it crosses (an opening's top and bottom 0.15 m are its frame, where a sill or lintel
             course belongs), and what is left shorter than 0.8 m is dropped. Two bands at the same height over the same
             stretch are one.
  parapets   a parapet's top is level or stepped: an outline with long sloping runs is a gable read off the photograph,
             which only a gable end of the roof can carry (the roof's own planes cut a gable wall, pipeline/facades.py),
             so it is dropped here, and kept as `gableOutline` when it peaks inside the wall (evidence the roof behind
             must turn its gable to this wall). A parapet more than 3 m above the wall's lowest top is dropped (the
             generator's photographed-parapet cap).
  reliefs    a row of three or more thin, tall reliefs over the same height (timber posts read one by one) becomes one
             timber frame (`timbering`: the posts, and rails along their foot and head); an applied ornament narrower
             than 0.05 m or with no height is dropped.
  slivers    a balcony, canopy, fire escape, read roof, planting, hedge or street-edge piece under 0.4 m wide is dropped.
  coverage   a reading whose openings, bands, bays and panels all sit in one part of its frame (under 70 % of it), the rest
             blank, is reported (`partial`): the photograph's frame missed part of the wall; it is redrawn only by a
             reading of a frame that covers it (regen/sides.py from registered cameras).

`clean(profile)` returns (cleaned copy, issues). Profiles are fractions of their frame: x of frameU, z of heightRefM.
Usage: facade_lint.py <profiles.json> [--ids building ids JSON] [--out report.json]   (a scan: what the rules would do)"""
import copy, json, math, sys

SLOPE = math.tan(math.radians(20.))   # steeper than this over a long run: a rake, not a parapet's step


def _wh(p):
    fu = p.get('frameU') or [0., 10.]; W = max(float(fu[1]) - float(fu[0]), .5); H = max(float(p.get('heightRefM') or 10.), 1.)
    return W, H


def _rects(p, W, H):
    out = []
    for o in p.get('openings') or []:
        r = o.get('rect')
        if r and len(r) == 4: out.append((r[0] * W, r[1] * H, r[2] * W, r[3] * H))
    for g in p.get('groups') or []:
        r = g.get('region')
        if r and len(r) == 4: out.append((r[0] * W, r[1] * H, r[2] * W, r[3] * H))
    return out


def clean(profile):
    p = copy.deepcopy(profile); W, H = _wh(p); issues = []
    holes = _rects(p, W, H)
    # bands: cut round the openings they cross, drop slivers, merge duplicates
    bands = []
    for b in p.get('bands') or []:
        xr = b.get('xRange') or [0., 1.]; zc = float(b.get('height', 0.)) * H; t = float(b.get('thicknessM', .2))
        runs = [(float(xr[0]) * W, float(xr[1]) * W)]
        for x0, z0, x1, z1 in holes:
            if zc + t / 2 <= z0 + .15 or zc - t / 2 >= z1 - .15: continue       # at or beyond the frame: a sill or lintel course
            nr = []
            for a, e in runs:
                if e <= x0 - .05 or a >= x1 + .05: nr.append((a, e)); continue
                if a < x0 - .05: nr.append((a, x0 - .05))
                if e > x1 + .05: nr.append((x1 + .05, e))
            if nr != runs: issues.append({'rule': 'bands', 'what': f'band at {zc:.1f} m cut round an opening at x {x0:.1f}-{x1:.1f} m'})
            runs = nr
        for a, e in runs:
            if e - a < .8:
                issues.append({'rule': 'bands', 'what': f'band at {zc:.1f} m: {e - a:.2f} m piece dropped'}); continue
            q = dict(b); q['xRange'] = [round(a / W, 5), round(e / W, 5)]
            dup = next((d for d in bands if abs(float(d.get('height', 0)) * H - zc) < .25 and min(d['xRange'][1], q['xRange'][1]) - max(d['xRange'][0], q['xRange'][0]) > .5 * (q['xRange'][1] - q['xRange'][0])), None)
            if dup is not None:
                issues.append({'rule': 'bands', 'what': f'band at {zc:.1f} m duplicates one at {float(dup.get("height", 0)) * H:.1f} m: merged'})
                if float(q.get('thicknessM', 0)) > float(dup.get('thicknessM', 0)): bands[bands.index(dup)] = q
                continue
            bands.append(q)
    if p.get('bands') is not None: p['bands'] = bands
    # parapets: level or stepped tops only
    keep = []
    for pp in p.get('parapets') or []:
        top = [(float(x) * W, float(z) * H) for x, z in pp.get('top') or []]
        if len(top) < 2: continue
        rake = max((abs(z2 - z1) / abs(x2 - x1) for (x1, z1), (x2, z2) in zip(top, top[1:]) if abs(x2 - x1) > .6), default=0.)
        span = max(z for _, z in top) - min(z for _, z in top)
        if rake > SLOPE:
            issues.append({'rule': 'parapets', 'what': f'parapet with a rake {math.degrees(math.atan(rake)):.0f} deg (a gable outline) dropped'})
            # a peak inside the outline with the top falling a metre or more to either side: the photograph saw a gable
            # end on this wall, which the roof behind it must make (pipeline/roof_forms.py gable_edges)
            zs = [z for _, z in top]; k = zs.index(max(zs))
            if 0 < k < len(top) - 1 and max(zs) - zs[0] >= 1. and max(zs) - zs[-1] >= 1.: p['gableOutline'] = pp.get('top')
            continue
        if span > 3.: issues.append({'rule': 'parapets', 'what': f'parapet varying {span:.1f} m dropped'}); continue
        keep.append(pp)
    if p.get('parapets') is not None: p['parapets'] = keep
    # slivers: a balcony, canopy, fire escape, read roof, planting, hedge or street-edge piece under 0.4 m wide is not one
    # (a reading moved onto its wall can leave a piece whose photograph showed only its edge)
    for k in ('balconies', 'canopies', 'fireEscapes', 'roofs', 'planting', 'hedges', 'streetEdge', 'porches'):
        if not p.get(k): continue
        keep_ = [q for q in p[k] if not (isinstance(q.get('x'), list) and len(q['x']) == 2 and (q['x'][1] - q['x'][0]) * W < .4)]
        if len(keep_) < len(p[k]): issues.append({'rule': 'slivers', 'what': f'{len(p[k]) - len(keep_)} {k} under 0.4 m wide dropped'})
        p[k] = keep_
    # a gable roof read on the wall (its bargeboards and overhang over most of the wall) is the same evidence as a gable
    # outline: the roof behind turns its gable to this wall, and the massing draws it (facade_detail skips the read one)
    for rf in p.get('roofs') or []:
        if rf.get('kind') == 'gable' and not p.get('gableOutline') and rf.get('x') and rf['x'][1] - rf['x'][0] >= .4:
            z = float(rf.get('z', 0.)); rise = float(rf.get('riseM', 0.)) / H
            if rise * H >= 1.: p['gableOutline'] = [[rf['x'][0], z], [(rf['x'][0] + rf['x'][1]) / 2, z + rise], [rf['x'][1], z]]
    # a row of three or more thin, tall reliefs over the same height (timber posts read one by one) is a timber frame:
    # posts where they were measured, tied by a rail along their foot and their head (facade_detail draws it)
    if p.get('reliefs'):
        post = lambda r: (r['rect'][2] - r['rect'][0]) * W <= .35 and (r['rect'][3] - r['rect'][1]) * H >= .8
        rows, rest = [], []
        for r in sorted(p['reliefs'], key=lambda r: r['rect'][0]):
            if not post(r): rest.append(r); continue
            row = next((w for w in rows if abs(w[0]['rect'][1] - r['rect'][1]) * H < .6 and abs(w[0]['rect'][3] - r['rect'][3]) * H < 1.2), None)
            (row.append(r) if row is not None else rows.append([r]))
        frames = []
        for w in rows:
            if len(w) < 3: rest += w; continue
            xs = [(r['rect'][0] + r['rect'][2]) / 2 * W for r in w]; gaps = [b - a for a, b in zip(xs, xs[1:])]
            if min(gaps) < .3 or max(gaps) > 3. * max(min(gaps), .3): rest += w; continue
            frames.append({'posts': [[r['rect'][0], r['rect'][2]] for r in w], 'rect': [min(r['rect'][0] for r in w), min(r['rect'][1] for r in w),
                           max(r['rect'][2] for r in w), max(r['rect'][3] for r in w)], 'widthM': round(sum((r['rect'][2] - r['rect'][0]) * W for r in w) / len(w), 3),
                           'projectionM': w[0].get('projectionM', .05), 'material': w[0].get('material')})
            issues.append({'rule': 'reliefs', 'what': f'{len(w)} posts read one by one joined into a timber frame'})
        if frames: p['timbering'] = (p.get('timbering') or []) + frames; p['reliefs'] = rest
    # reliefs with no size
    if p.get('reliefs'):
        rl = [r for r in p['reliefs'] if (r['rect'][2] - r['rect'][0]) * W >= .05 and (r['rect'][3] - r['rect'][1]) * H >= .05]
        if len(rl) < len(p['reliefs']): issues.append({'rule': 'reliefs', 'what': f'{len(p["reliefs"]) - len(rl)} relief(s) with no size dropped'})
        p['reliefs'] = rl
    # coverage: where the reading's parts are, across its frame
    xs = [(o['rect'][0], o['rect'][2]) for o in p.get('openings') or [] if o.get('rect')] + [(g['region'][0], g['region'][2]) for g in p.get('groups') or [] if g.get('region')]
    xs += [tuple(b['xRange']) for b in p.get('bands') or [] if b.get('xRange')] + [(b['rect'][0], b['rect'][2]) for b in p.get('bays') or [] if b.get('rect')]
    if xs:
        lo, hi = max(0., min(a for a, _ in xs)), min(1., max(e for _, e in xs))
        if hi - lo < .7 and (lo > .25 or hi < .75):
            issues.append({'rule': 'coverage', 'what': f'parts only over {lo:.2f}-{hi:.2f} of the frame ({(hi - lo) * W:.1f} of {W:.1f} m)', 'partial': [round(lo, 3), round(hi, 3)]})
            # a side the photographs barely saw (its parts over under a quarter of it) gives its colours, not its layout: the
            # wall takes the building's inferred openings like an unseen wall, not a blank wall with a scrap at one end
            if hi - lo < .25: p['materialOnly'] = True; issues.append({'rule': 'coverage', 'what': 'under a quarter seen: colours only'})
    return p, issues


if __name__ == '__main__':
    import argparse, collections
    ap = argparse.ArgumentParser(); ap.add_argument('profiles'); ap.add_argument('--ids'); ap.add_argument('--out'); a = ap.parse_args()
    ids = set(json.load(open(a.ids))) if a.ids else None
    rep = {}; cnt = collections.Counter(); sides = collections.Counter()
    for pr in json.load(open(a.profiles))['profiles']:
        bid = str((pr.get('match') or {}).get('featureId') or pr['id'].replace('regen-', '').split('__')[0])
        if ids is not None and bid not in ids and bid.split('_p')[0] not in ids: continue
        _, iss = clean(pr)
        if iss:
            rep[pr['id']] = iss
            for r in {i['rule'] for i in iss}: sides[r] += 1
            for i in iss: cnt[i['rule']] += 1
    if a.out: json.dump(rep, open(a.out, 'w'), indent=1)
    print(json.dumps({'profilesWithIssues': len(rep), 'issues': dict(cnt), 'sidesByRule': dict(sides)}))
