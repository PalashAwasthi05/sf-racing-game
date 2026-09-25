"""Render every user-flagged view (work/review/flagged-views.json) for a candidate and the live release from the same
cameras, next to the photograph where the view names a panorama, and list what changed in the flagged buildings'
evidence (parts, wall sources, materials). Writes <candidate>/flagged-views/<view>__<camera>.jpg and index.json;
publish_full_candidate_v24.py refuses a candidate whose index does not cover the current GLB and every view.
Usage: check_flagged_views.py --candidate work/<dir> [--live work/<dir>]"""
import argparse, hashlib, json, pathlib, sys, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import review_render as R
import frontage_crops_v15 as fc

ROOT = pathlib.Path(__file__).resolve().parent
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''): h.update(block)
    return h.hexdigest()


def glb_of(d):
    for p in (d / 'materials/candidate.glb', d / 'final/candidate.glb'):
        if p.exists(): return p
    raise SystemExit(f'no GLB in {d}')


def live_dir():
    gen = json.load(open(ROOT / 'king-street/public/generation.json'))
    return pathlib.Path(gen['candidate']).resolve().parent.parent if gen.get('candidate') else None


def evidence(d):
    p = d / 'king-street-evidence.json'
    if p.exists():
        e = json.load(open(p)); return e['facadeWalls'], e['world']['buildings']
    e = json.load(open(d / 'public/facade-evidence.json')); return e['walls'], json.load(open(d / 'public/world.json'))['buildings']


def wall_key(w):
    return (w['buildingId'], round(w['midpoint'][0], 0), round(w['midpoint'][1], 0), round(float(np.degrees(np.arctan2(w['normal'][1], w['normal'][0]))) / 5) * 5)


def describe(w):
    src = 'reading ' + str(w.get('grammarId') or w.get('archetype')) if (w.get('edge') == 'grammar' or w.get('grammarId')) else str(w.get('status'))
    return f"{src} | {w.get('material')} | h {w['heightM']:.1f} | {w.get('exposure') or ''}"


def changes(ids, live, cand):
    lw, lb = live; cw, cb = cand; out = []
    keep = lambda b: any(b == i or b.startswith(i + '_p') for i in ids)
    lparts = sorted((b['id'], b.get('height')) for b in lb if keep(b['id'])); cparts = sorted((b['id'], b.get('height')) for b in cb if keep(b['id']))
    if lparts != cparts: out.append(f'parts: {lparts} -> {cparts}')
    L = {wall_key(w): w for w in lw if keep(w['buildingId'])}; C = {wall_key(w): w for w in cw if keep(w['buildingId'])}
    for k in sorted(set(L) | set(C), key=str):
        a, b = L.get(k), C.get(k)
        if a and b and describe(a) != describe(b): out.append(f'wall {k[0]} at {k[1]:.0f},{k[2]:.0f} facing {k[3]:.0f} deg, {b["lengthM"]:.1f} m: {describe(a)} -> {describe(b)}')
        elif a and not b: out.append(f'wall removed {k[0]} at {k[1]:.0f},{k[2]:.0f} facing {k[3]:.0f} deg ({a["lengthM"]:.1f} m, {describe(a)})')
        elif b and not a: out.append(f'wall added {k[0]} at {k[1]:.0f},{k[2]:.0f} facing {k[3]:.0f} deg ({b["lengthM"]:.1f} m, {describe(b)})')
    return out


def sheet(tiles, view, camera):
    w, h = R.WIDTH, R.HEIGHT; font = ImageFont.truetype(FONT, 22); small = ImageFont.truetype(FONT, 17)
    lines = [f"{view['id']} / {camera['name']}   reported {', '.join(view['reported'])}"]
    for d in view.get('defects', []):
        lines += textwrap.wrap(f"[{d['status']}] {d['text']}", 150)
    top = 40 + 22 * len(lines)
    out = Image.new('RGB', (len(tiles) * (w + 10), h + top), (255, 255, 255)); dr = ImageDraw.Draw(out)
    for k, s in enumerate(lines): dr.text((8, 6 + 22 * k), s, fill=(30, 30, 30), font=small if k else font)
    for i, (label, im) in enumerate(tiles):
        out.paste(im, (i * (w + 10), top)); dr.rectangle([i * (w + 10), top, i * (w + 10) + 14 + 12 * len(label), top + 32], fill=(0, 0, 0))
        dr.text((i * (w + 10) + 7, top + 5), label, fill=(255, 255, 255), font=font)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--candidate', type=pathlib.Path, required=True); ap.add_argument('--live', type=pathlib.Path)
    a = ap.parse_args(); cand = a.candidate.resolve(); live = (a.live.resolve() if a.live else live_dir())
    reg = json.load(open(ROOT / 'review/flagged-views.json')); out = cand / 'flagged-views'; out.mkdir(exist_ok=True)
    world = json.load(open(cand / 'public/world.json')); frame = fc.LocalFrame(world['center']['lon'], world['center']['lat'])
    cand_glb = glb_of(cand); renders = {}
    for label, d in (('live', live), ('candidate', cand)):
        if d is None: continue
        b = R.load(str(glb_of(d)))
        for v in reg['views']:
            for c in v['cameras']: renders[(label, v['id'], c['name'])] = R.render(b, c['from'], c['at'])
        R.free(b)
    index = {'candidate': str(cand), 'candidateGlbSha256': sha(cand_glb), 'live': str(live) if live else None, 'views': []}
    ev_c = evidence(cand); ev_l = evidence(live) if live else None
    for v in reg['views']:
        rec = {'id': v['id'], 'sheets': [], 'openDefects': [d['id'] for d in v.get('defects', []) if d['status'] != 'fixed'],
               'evidenceChanges': changes(v['buildings'], ev_l, ev_c) if ev_l else []}
        for c in v['cameras']:
            tiles = []
            if c.get('panorama'): tiles.append(('photograph', R.photo(c['panorama'], c['from'], c['at'], frame)))
            if ('live', v['id'], c['name']) in renders: tiles.append(('live', renders[('live', v['id'], c['name'])]))
            tiles.append(('candidate', renders[('candidate', v['id'], c['name'])]))
            p = out / f"{v['id']}__{c['name']}.jpg"; sheet(tiles, v, c).save(p, quality=88); rec['sheets'].append(str(p))
        index['views'].append(rec)
        print(f"{v['id']}: {len(rec['sheets'])} sheets, open defects {rec['openDefects']}, evidence changes {len(rec['evidenceChanges'])}")
        for s in rec['evidenceChanges'][:40]: print('   ', s)
    if reg.get('pending'): print(f"{len(reg['pending'])} flagged reports still have no pinned camera (see review/flagged-views.json 'pending').")
    (out / 'index.json').write_text(json.dumps(index, indent=1)); print('wrote', out / 'index.json')


if __name__ == '__main__':
    main()
