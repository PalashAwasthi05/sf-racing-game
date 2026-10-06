"""Simple roof forms fitted to a building part's measured roof surface, for the outlines whose returns the general plane
fit (general_world.roof_planes) leaves flat: a house's gable or hip is two to four planes of 20-80 m2, too few cells
each for that fit to validate, and the part was drawn as a flat box at its eaves.

The outline's rectangle gives the eaves. A form is a set of eave edges, each rising inward at one shared pitch, so the
roof is z = eave + pitch x (distance to the nearest of them), optionally capped flat at a ridge or plateau. As planes
of a 'min' model (the model roof_planes.roof_solid builds and the walls follow):
  shed   one eave;
  gable  two opposite eaves (the other two sides are gable ends, drawn to the rake by the facade compiler);
  hip    all four;
  twin   two gables side by side across the short way (an M roof): each half rises from its outer eave and from the
         valley down the middle, two patches.
A form is kept only when it holds clearly more returns than the best flat level does: a flat roof with a
parapet, plant or a penthouse stays flat. The fit is robust (a line held by the most returns, refined on them), so
a false front, a chimney or a dormer does not tilt it. Location-free: nothing here names a place or a style."""
import numpy as np
from shapely import contains_xy

FORMS = {'shed': ([0], [1], [2], [3]), 'gable': ([0, 2], [1, 3]), 'hip': ([0, 1, 2, 3],), 'twin': ([0, 2], [1, 3])}


def _eaves(rect):
    """The rectangle's four edges as (a point on it, its inward unit normal, its length)."""
    xy = np.asarray(rect.exterior.coords)[:4]; c = xy.mean(0); out = []
    for i in range(4):
        a, b = xy[i], xy[(i + 1) % 4]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]])
        out.append((a, n if (c - a) @ n > 0 else -n, float(np.linalg.norm(b - a))))
    return out


def _robust(D, z, tol, cap=None, tries=300, seed=812):
    """eave, pitch of z ~ eave + pitch*D (capped at `cap`, if given) held by the most returns within `tol`: pairs of
    returns propose a line, the one holding the most is refined on its own returns. A false front, side parapets or a
    lower porch ring can be half the returns; they are left out, not averaged in. Seeded: a build is repeatable."""
    rng = np.random.default_rng(seed); n = len(z)
    below = np.ones(n, bool) if cap is None else z < cap - tol
    idx = np.where(below)[0]
    if len(idx) < 30: return None
    best, best_count = None, 0
    for _ in range(tries):
        i, j = rng.choice(idx, 2, replace=False)
        if abs(D[j] - D[i]) < 1.: continue
        s = (z[j] - z[i]) / (D[j] - D[i]); e = z[i] - s * D[i]
        pred = e + s * D if cap is None else np.minimum(e + s * D, cap); count = int((np.abs(z - pred) < tol).sum())
        if count > best_count: best, best_count = (e, s), count
    if best is None: return None
    e, s = best
    for _ in range(3):
        pred = e + s * D if cap is None else np.minimum(e + s * D, cap); m = (np.abs(z - pred) < tol) & (e + s * D < (np.inf if cap is None else cap))
        if m.sum() < 30: return None
        (e, s), *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), D[m]], z[m], rcond=None)
    pred = e + s * D if cap is None else np.minimum(e + s * D, cap)
    return float(e), float(s), np.abs(z - pred)


def _flat(z, tol):
    """The flat roof the returns support best: the level holding the most of them within `tol` (a parapet or plant
    standing on the roof is not averaged into it), its residuals."""
    zs = np.sort(z); counts = np.searchsorted(zs, zs + 2 * tol) - np.arange(len(zs)); k = int(np.argmax(counts))
    level = float(np.median(zs[k:k + counts[k]])); return np.abs(z - level)


PREFERENCE = ('gable', 'hip', 'twin', 'shed')


def _rank(c):
    """Order among forms that hold about as many returns: uncapped before capped, then PREFERENCE."""
    return (c['cap'] is not None, PREFERENCE.index(c['form']))


def _score(r, tol):
    """(median absolute residual, share of returns within `tol`)."""
    return float(np.median(r)), float((r < tol).mean())


def fit(footprint, points, min_slope=.15, max_slope=2.5, min_rise=1., tol=.3, rect_share=.75, gain=.5, min_inliers=.55, min_lead=.15, max_points=6000):
    """([patch], report): a patch {'geometry', 'model', 'form'} when a pitched form explains the roof, else ([], report).
    `points` are the part's roof surface returns (x, y, z in the generator's frame). A form must hold `min_inliers` of
    the returns within `tol`, `min_lead` more of them than a flat roof does, and halve the flat roof's median residual."""
    if footprint.geom_type != 'Polygon':
        # a municipal outline is often a one-part multipolygon; slivers left by a clip or a split keep the flat cap
        polys = [q for q in getattr(footprint, 'geoms', []) if q.geom_type == 'Polygon']
        main = max(polys, key=lambda q: q.area) if polys else None
        if main is None or main.area < .9 * footprint.area: return [], {'status': 'not-one-outline'}
        footprint = main
    rect = footprint.minimum_rotated_rectangle
    if rect.geom_type != 'Polygon' or footprint.area < rect_share * rect.area: return [], {'status': 'not-rectangular'}
    p = np.asarray(points)[:, :3]; p = p[contains_xy(footprint, p[:, 0], p[:, 1])]
    if len(p) < 60: return [], {'status': 'too-few-returns'}
    if len(p) > max_points: p = p[np.linspace(0, len(p) - 1, max_points).astype(int)]
    xy, z = p[:, :2], p[:, 2]; E = _eaves(rect)
    d = np.stack([(xy - a) @ n for a, n, _ in E], 1)
    flat_med, flat_in = _score(_flat(z, tol), tol)
    report = {'status': 'flat', 'flatMedianM': round(flat_med, 3), 'flatInliers': round(flat_in, 3)}
    best = None
    for form, sets in FORMS.items():
        for S in sets:
            # a twin's valley runs midway between its two eaves: the distance to it counts like an eave's
            W = E[S[0] + 1][2] if form == 'twin' else None
            D = d[:, list(S)].min(1) if form != 'twin' else np.minimum(d[:, list(S)].min(1), np.abs(d[:, S[0]] - W / 2))
            f = _robust(D, z, tol)
            if f is None: continue
            e, s, r = f
            rise = s * float(np.quantile(D, .98))
            if not (min_slope <= s <= max_slope) or rise < min_rise: continue
            med, inl = _score(r, tol)
            cand = {'form': form, 'eaves': list(S), 'eaveZ': e, 'pitch': s, 'riseM': rise, 'medianM': med, 'inliers': inl, 'cap': None}
            # a flat top over the highest part (a ridge cap, a truncated hip): kept when it holds clearly more returns
            top = e + s * D
            for q in (.6, .7, .8, .9):
                cap = float(np.quantile(top, q)); g = _robust(D, z, tol, cap=cap)
                if g is None: continue
                ce, cs, cr = g; cmed, cin = _score(cr, tol)
                if min_slope <= cs <= max_slope and cap - ce >= min_rise and cin >= cand['inliers'] + .05:
                    cand.update(eaveZ=ce, pitch=cs, medianM=cmed, inliers=cin, cap=cap, riseM=cap - ce)
            # within 3 % of the returns the commoner form wins: a gable over a hip (whose end planes must earn their place),
            # and either over one slope rising to a flat top, which also fits half of a ridged roof
            if best is None or cand['inliers'] > best['inliers'] + .03 or (abs(cand['inliers'] - best['inliers']) <= .03 and _rank(cand) < _rank(best)): best = cand
    if best is None: return [], report
    report.update(best={k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
    if best['inliers'] < min_inliers or best['inliers'] < flat_in + min_lead or best['medianM'] > gain * flat_med: return [], report
    s, e = best['pitch'], best['eaveZ']; cap = [[0., 0., best['cap']]] if best['cap'] is not None else []
    rise = lambda i: [s * float(E[i][1][0]), s * float(E[i][1][1]), e - s * float(E[i][1] @ E[i][0])]   # from eave i inward
    name = best['form'] + ('-capped' if best['cap'] is not None else ''); report['status'] = 'form'
    if best['form'] != 'twin':
        return [{'geometry': footprint, 'model': {'planes': [rise(i) for i in best['eaves']] + cap, 'operation': 'min'}, 'form': name}], report
    # each half: its outer eave's plane, and the plane rising from the valley toward it (e + s*(W/2 - d_i))
    from shapely.geometry import Polygon
    W = E[best['eaves'][0] + 1][2]; out = []
    for i in best['eaves']:
        a, n = E[i][0], E[i][1]; u = np.array([n[1], -n[0]]); L = E[i][2] + 2.   # along the eave, either way
        half = Polygon([a - u * L, a + u * L, a + u * L + n * W / 2, a - u * L + n * W / 2]).intersection(footprint)
        if half.is_empty: continue
        valley = [-s * float(n[0]), -s * float(n[1]), e + s * W / 2 + s * float(n @ a)]
        out.append({'geometry': half, 'model': {'planes': [rise(i), valley] + cap, 'operation': 'min'}, 'form': name})
    return out, report
