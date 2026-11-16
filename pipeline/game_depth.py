"""Game depth: the measured facade relief, deepened so it reads at play distance.

The photo readings give true depths, and most of this city's walls are shallow: glass 10-15 cm behind the face, sills
and string courses standing 5-20 cm proud. Under a high sun those cast shadows a few centimetres tall, and from a moving
car a facade reads as a printed sheet. The user chose (2026-11-13) to exaggerate them, as games do. This is a stated
departure from measured depth for looks, applied uniformly to every reading; it adds no element a reading does not
name, and it changes no position, size or material in the wall's plane.

Applied once, to the readings as the generator loads them. `DEPTH` 1 (environment SFW_GAME_DEPTH=1) turns it off.
"""
import os
DEPTH = float(os.environ.get("SFW_GAME_DEPTH", "3"))
# caps keep a deepened element plausible and clear of the wall's 0.3 m slab and the room backing behind the glass
CAPS = {'recessM': .5, 'surround': .15, 'hood': .45, 'bandProjection': .42, 'bandDepth': .7, 'pier': .45, 'relief': .18}
# the sill every window is given (facade_grammar): projection from the wall face
SILL_PROJECTION_M = .19 * (1 + (DEPTH - 1) / 2)


def _scale(d, key, cap, factor=DEPTH):
    if isinstance(d, dict) and isinstance(d.get(key), (int, float)):
        d[key] = min(cap, float(d[key]) * factor) if d[key] < cap else float(d[key])


def _style(st):
    if not isinstance(st, dict): return
    _scale(st, 'recessM', CAPS['recessM'])
    _scale(st.get('surround'), 'projectionM', CAPS['surround'])
    _scale(st.get('hood'), 'projectionM', CAPS['hood'])


def deepen(profiles):
    """Deepen every profile in place; returns how many values changed."""
    if DEPTH == 1: return 0
    before = repr(profiles)
    for p in profiles:
        for g in p.get('groups', []) or []: _style(g.get('style'))
        for o in p.get('openings', []) or []: _style(o.get('style'))
        for b in p.get('bands', []) or []:
            _scale(b, 'projectionM', CAPS['bandProjection']); _scale(b, 'depthM', CAPS['bandDepth'], 1 + (DEPTH - 1) / 2)
        for pr in p.get('piers', []) or []: _scale(pr, 'projectionM', CAPS['pier'])
        for key in ('reliefs', 'trims', 'ornaments'):
            for r in p.get(key, []) or []: _scale(r, 'projectionM', CAPS['relief'])
    return int(before != repr(profiles))
