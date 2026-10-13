"""Do a side's two photographs agree on where its wall is? The rectified crops of one side from two cameras show the same
wall in the same frame when both cameras are placed right; a camera metres off shifts its crop along the wall. Per side
with two views: the offset (metres, along and up the wall) between the two crops by phase correlation of their edge
images, for two sets of crops (the provider poses' and the registered cameras'). A registration worth using brings the
two views together. Independent of the skyline fit itself: it compares two cameras against each other.
Usage: views_agree.py <sides dir A> <sides dir B> [--ids ids.json] [--out out.json]   (run from work/)"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image


def edges(path, pad):
    a = np.asarray(Image.open(path).convert('L'), np.float32); a = a[pad:a.shape[0] - pad, pad:a.shape[1] - pad]
    gy, gx = np.gradient(a); e = np.hypot(gx, gy); e -= e.mean(); return e / (e.std() + 1e-6)


def shift(a, b):
    """(dx, dy) pixels moving b onto a, by phase correlation (a Hann window against the edges' wrap-around)."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]); a, b = a[:h, :w], b[:h, :w]
    win = np.outer(np.hanning(h), np.hanning(w)); A = np.fft.fft2(a * win); B = np.fft.fft2(b * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9; r = np.fft.ifft2(R).real
    y, x = np.unravel_index(int(np.argmax(r)), r.shape)
    if y > h // 2: y -= h
    if x > w // 2: x -= w
    peak = float(r.max()) / (float(np.abs(r).mean()) + 1e-9)
    return float(x), float(y), peak


def measure(d, ids):
    pieces = {}
    for f in sorted(Path(d).glob('sides*.json')):                 # a sides folder written in shards, or merged
        for q in json.load(open(f))['pieces']:
            if q.get('views') and (q['id'] not in pieces or len(q['views']) > len(pieces[q['id']]['views'])): pieces[q['id']] = q
    pieces = list(pieces.values()); out = {}
    for p in pieces:
        if ids is not None and p['building'] not in ids and p['building'].split('_p')[0] not in ids: continue
        if len(p.get('views') or []) < 2: continue
        v0, v1 = p['views'][:2]
        try: a, b = edges(Path(d) / 'img' / v0['rect'], v0['rectPad']), edges(Path(d) / 'img' / v1['rect'], v1['rectPad'])
        except Exception: continue
        dx, dy, pk = shift(a, b); ppm = float(v0['ppm'])
        out[p['id']] = {'dxM': round(dx / ppm, 2), 'dyM': round(dy / ppm, 2), 'peak': round(pk, 1), 'images': [v0['image'], v1['image']]}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--ids'); ap.add_argument('--out'); x = ap.parse_args()
    ids = set(json.load(open(x.ids))) if x.ids else None
    A, B = measure(x.a, ids), measure(x.b, ids)
    common = [k for k in A if k in B and A[k]['images'] == B[k]['images'] and min(A[k]['peak'], B[k]['peak']) > 6.]   # same two cameras, a clear match in both
    da = np.array([np.hypot(A[k]['dxM'], A[k]['dyM']) for k in common]); db = np.array([np.hypot(B[k]['dxM'], B[k]['dyM']) for k in common])
    res = {'sides': len(common), 'offsetMedianA': round(float(np.median(da)), 2) if len(da) else None, 'offsetMedianB': round(float(np.median(db)), 2) if len(db) else None,
           'p90A': round(float(np.percentile(da, 90)), 2) if len(da) else None, 'p90B': round(float(np.percentile(db, 90)), 2) if len(db) else None,
           'closerInB': round(float(np.mean(db < da)), 2) if len(da) else None}
    if x.out: json.dump({'summary': res, 'A': A, 'B': B}, open(x.out, 'w'), indent=1)
    print(json.dumps(res))
