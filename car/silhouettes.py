"""Compare the body's side, plan and front silhouettes with the measured outlines; writes a PNG sheet."""
import sys, time, json
import numpy as np
from PIL import Image, ImageDraw
import roadster as R
from mesher import grid_axes, sample

def run(out, h=.01):
    t0 = time.time()
    axes = grid_axes(*R.BOUNDS, h)
    F = sample(R.body, axes)
    xs, ys, zs = axes
    inside = F < 0
    side = inside.any(2)      # (x, y)
    plan = inside.any(1)      # (x, z)
    front = inside.any(0)     # (y, z)
    print('sampled', F.shape, '%.1fs' % (time.time() - t0))
    ref = R.REF
    rx = np.array(ref['x']); rx = np.where(rx > 1.5, 1.5 + (rx - 1.5) * (R.NOSE - 1.5) / (rx.max() - 1.5), rx)
    S = 400  # px per metre
    W, H = int(4.6 * S), int(1.4 * S + 1.0 * S + 1.4 * S)
    img = Image.new('RGB', (W, H), (250, 250, 250)); d = ImageDraw.Draw(img)
    X = lambda x: (x + 2.25) * S
    # side view: body silhouette filled, measured top/bottom as red polyline
    for i, x in enumerate(xs):
        col = np.nonzero(side[i])[0]
        if len(col):
            d.line([(X(x), (1.35 - ys[col.max()]) * S), (X(x), (1.35 - ys[col.min()]) * S)], fill=(150, 170, 190))
    top = np.array(ref['top']) - R.DROP; bot = np.array(ref['bottom']) - R.DROP
    d.line([(X(a), (1.35 - b) * S) for a, b in zip(rx, top)], fill=(220, 30, 30), width=2)
    d.line([(X(a), (1.35 - b) * S) for a, b in zip(rx, bot)], fill=(220, 30, 30), width=2)
    for xa, r in ((R.AXLE, R.R_FRONT), (-R.AXLE, R.R_REAR)):
        d.ellipse([X(xa - r), (1.35 - 2 * r) * S, X(xa + r), 1.35 * S], outline=(40, 40, 40), width=2)
    # plan view
    oy = 1.45 * S
    for i, x in enumerate(xs):
        col = np.nonzero(plan[i])[0]
        if len(col):
            d.line([(X(x), oy + (1.0 + zs[col.min()]) * S * .5), (X(x), oy + (1.0 + zs[col.max()]) * S * .5)], fill=(150, 170, 190))
    half = np.array(ref['half'])
    d.line([(X(a), oy + (1.0 + b) * S * .5) for a, b in zip(rx, half)], fill=(220, 30, 30), width=2)
    d.line([(X(a), oy + (1.0 - b) * S * .5) for a, b in zip(rx, half)], fill=(220, 30, 30), width=2)
    # front view (max over x)
    ox, oy2 = 1.0 * S, 2.5 * S
    for j, y in enumerate(ys):
        col = np.nonzero(front[j])[0]
        if len(col):
            d.line([(ox + (1.0 + zs[col.min()]) * S, oy2 + (1.35 - y) * S), (ox + (1.0 + zs[col.max()]) * S, oy2 + (1.35 - y) * S)], fill=(150, 170, 190))
    fh = np.array(ref['front_view_h']) - R.DROP; fw = np.array(ref['front_view_half'])
    d.line([(ox + (1 + w) * S, oy2 + (1.35 - hh) * S) for hh, w in zip(fh, fw)], fill=(220, 30, 30), width=2)
    d.line([(ox + (1 - w) * S, oy2 + (1.35 - hh) * S) for hh, w in zip(fh, fw)], fill=(220, 30, 30), width=2)
    img.save(out)
    # numbers: silhouette top vs measured top
    errs = []
    for a, b in zip(rx, top):
        i = np.argmin(np.abs(xs - a)); col = np.nonzero(side[i])[0]
        if len(col): errs.append((a, ys[col.max()] - b))
    e = np.array(errs)
    print('side top error: median %.3f  p90 %.3f' % (np.median(np.abs(e[:, 1])), np.percentile(np.abs(e[:, 1]), 90)))
    return F, axes

if __name__ == '__main__':
    run(sys.argv[1])
