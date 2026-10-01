"""Photo labels for scoring, once per view, in the wall frame (12 px/m): what is opening, what is wall, what cannot be
judged (trees, cars, sky, signs). SegFormer for occluders and sky; Grounding DINO for windows, doors, signs and awnings
(tiled, as the judge does); whole-wall glazing (curtain walls) counted as opening when the detector is confident, left
unjudged when it is unsure, so a scorer neither rewards deleting real glass nor drawing glass on a dark wall.
Near-black wall pixels (glass or shade) are left unjudged too. Cached as regen/v1/codes/<side>_<view>.npz."""
import argparse, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import aligncorr, score_sides as S, judge_facades_v27 as J
from facade_segment_v27 import Segmenter, Detector, glazed_boxes
PPM = 12.
GLAZING_OPENING, GLAZING_UNSURE = .45, .35

def codes_for(u8, seg, det):
    codes = seg.labels(u8).copy(); codes[codes == 1] = 0; codes[codes == 2] = 0
    po, _, _ = J.photo_openings(u8.astype(float))
    H, W = codes.shape
    for x0, y0, x1, y1, kind, sc in det.boxes(u8):
        sl = (slice(int(max(0, y0)), int(min(H, y1))), slice(int(max(0, x0)), int(min(W, x1))))
        if kind in ('window', 'door') and (x1 - x0) * (y1 - y0) / PPM ** 2 <= 40: codes[sl] = np.where(codes[sl] == 3, 3, 1)
        elif kind in ('sign', 'awning'): codes[sl] = np.where(codes[sl] == 0, 2, codes[sl])
    codes[po & (codes == 0)] = 1
    for x0, y0, x1, y1, sc in glazed_boxes(det, u8):
        sl = (slice(int(max(0, y0)), int(min(H, y1))), slice(int(max(0, x0)), int(min(W, x1))))
        if sc >= GLAZING_OPENING: codes[sl] = np.where(codes[sl] == 0, 1, codes[sl])
        elif sc >= GLAZING_UNSURE: codes[sl] = np.where(codes[sl] == 0, 3, codes[sl])
    # near-black pixels are glass or deep shade, and neither detector can tell which: not judged either way
    lum = u8.astype(float).mean(2); codes[(lum < 45) & (codes == 0)] = 3
    return codes

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ids', nargs='*'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    root = HERE / 'v1'; out = root / 'codes'; out.mkdir(exist_ok=True)
    byid = {p['id']: p for p in json.load(open(root / 'sides/sides.json'))['pieces']}; al = json.load(open(root / 'alignment.json'))
    ids = a.ids or sorted(f.stem for f in (root / 'final3').glob('*.json'))
    seg, det = Segmenter(), Detector(); t0 = time.time(); n = 0
    for k, i in enumerate(ids):
        p = byid.get(i)
        if not p: continue
        for vi, v in enumerate(p['views']):
            f = out / f'{i}_{vi}.npz'
            if f.exists(): continue
            ph = S.photo_in_wall_frame(p, v, aligncorr.choose((al.get(i) or {}).get(str(vi))), root / 'sides/img')
            np.savez_compressed(f, codes=codes_for(ph, seg, det)); n += 1
            if a.limit and n >= a.limit: print('done', n, 'limit'); sys.exit(0)
        if k % 20 == 0: print(k, len(ids), n, round(time.time() - t0), 's', flush=True)
    print('done', n, round(time.time() - t0), 's')
