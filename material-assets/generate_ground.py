"""Generate the tiling close-range ground layers the viewer blends on roads and pavements (app/ground-detail.ts).

Eight 1024 x 1024 layers stacked top to bottom in one RGB image; the first six are 1 m squares at 1 mm a texel:
  0 asphalt, dense: small aggregate in dark binder (recent surfacing)
  1 asphalt, worn: larger pale stones standing proud of eroded binder (old surfacing)
  2 asphalt, cracked: dense asphalt broken by a crack network with ravelled edges
  3 concrete, broomed: fine ridges one way, pits, sand grain
  4 concrete, weathered: the paste worn off sand and small stones, pits
  5 concrete, trowelled: smooth, faint float arcs, a few pits (a recent pour)
and two are 8 m squares for what still reads from a moving car:
  6 old asphalt: tar-sealed and open cracks, mottling, stains
  7 old concrete: stains, long cracks, worn patches
R, G: the surface normal's x, y (0.5 = flat); B: tone, the albedo's brightness about 0.5.
A second image, <out>-height.png, holds each layer's height (white = the surface's high points).
Everything wraps: noise is filtered in the frequency domain and cell patterns use a periodic nearest-neighbour search.
Run: uv run --python 3.11 --with numpy --with scipy --with pillow python generate_ground.py <out.png>
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

N = 1024
rng = np.random.default_rng(718)
yy, xx = np.mgrid[0:N, 0:N]
grid = np.stack([xx.ravel() + .5, yy.ravel() + .5], 1)
fx = np.fft.fftfreq(N)[None, :] * N
fy = np.fft.fftfreq(N)[:, None] * N
radial = np.hypot(fx, fy)


def noise(lo, hi, slope=1., stretch=1.):
    """Band-limited noise between `lo` and `hi` cycles a tile, 1/f^slope, mean 0, deviation 1; `stretch` > 1 draws it out along x."""
    r = np.hypot(fx * stretch, fy)
    amp = np.where((r >= lo) & (r <= hi), 1. / np.maximum(r, 1.) ** slope, 0.)
    out = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * amp).real
    return out / out.std()


def blur(a, sigma):
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (np.pi * sigma * radial / N) ** 2)).real


def cells(count, k=2):
    """Periodic Voronoi: distance to the nearest and second nearest of `count` points, the nearest's index and a value per point."""
    pts = rng.random((count, 2)) * N
    d, i = cKDTree(pts, boxsize=N).query(grid, k=k)
    return d[:, 0].reshape(N, N), d[:, 1].reshape(N, N), i[:, 0].reshape(N, N), rng.random(count)


def stones(count, gap, dome=.5):
    """Aggregate: packed, rounded stones of varied size and grey, `gap` of binder between them (in stone sizes).
    Returns (height 0..1, mask, per-stone grey)."""
    d1, d2, idx, val = cells(count)
    size = np.sqrt(N * N / count)
    shrink = gap * (.5 + 2.2 * rng.random(count)[idx] ** 2)       # most stones touch their neighbours, some sit apart
    edge = blur((d2 - d1) / size, size * .12)                     # rounded corners; 0 on a cell boundary
    inside = np.clip((edge - shrink) * 5., 0, 1)
    return inside ** dome, inside > .08, val[idx]


def cracks(count, width):
    d1, d2, _, _ = cells(count)
    warp = (d2 - d1) + 2.2 * noise(6, 60, 1.2) + .8 * noise(60, 300, .5)
    return np.clip(1. - warp / width, 0, 1)


def pits(count, radius):
    d1, _, idx, val = cells(count, 2)
    r = radius * (.4 + 1.2 * val[idx])
    return np.clip(1. - d1 / r, 0, 1) ** .7


def asphalt(count, gap, pale, erosion):
    """Coarse stones with fine ones filling between them, in binder. `erosion`: how far the binder has worn below the stones."""
    h, mask, grey = stones(count, gap)
    fh, fmask, fgrey = stones(count * 7, gap * 1.3)
    fine = fmask & ~mask
    binder = noise(40, 400, .6)
    height = np.where(mask, 1. - erosion * .0 + 0 * h, 0.)
    height = np.where(mask, (1 - erosion) + erosion * h, np.where(fine, (1 - erosion) * .9 + erosion * .55 * fh, (1 - erosion) * .8 + .03 * binder))
    height = blur(height, .8) + .025 * noise(150, 500, 0)          # stone faces are not polished glass
    # stone colour: mostly mid grey, some pale quartz, a few dark basalt; binder near black
    tone = np.where(mask, (.3 + pale * grey ** 1.5) * (.6 + .4 * h), np.where(fine, (.26 + .7 * pale * fgrey ** 1.5) * (.6 + .4 * fh), .13 + .02 * binder))
    return height, blur(tone + .03 * noise(100, 500, 0), .6)


def layers():
    out = []
    h, t = asphalt(9000, .1, .3, .45); out.append((h, t, 1.6))
    h, t = asphalt(4200, .07, .5, .9)
    h -= .25 * pits(160, 9); t *= 1 - .5 * pits(160, 9)           # stones plucked out
    out.append((h, t, 2.6))
    h, t = asphalt(9000, .1, .3, .5)
    c = np.maximum(cracks(55, 5.), .7 * cracks(260, 3.))          # a main network and hairlines off it
    near = blur(c, 9)
    h = h * (1 - .5 * np.clip(near * 3, 0, 1)) - 1.3 * c           # ravelled, lower edges, then the gap
    t = t * (1 - .85 * c) * (1 - .25 * np.clip(near * 3, 0, 1))
    out.append((h, t, 2.2))
    # broomed concrete: ridges along x from stretched noise, sand, pits
    ridges = noise(50, 260, .1, stretch=18.)
    sand = noise(120, 500, 0)
    p = np.maximum(pits(420, 4.5), .8 * pits(60, 10))
    h = .16 * ridges + .05 * sand + .1 * noise(4, 40, 1.4) - .9 * p
    t = .5 + .05 * noise(2, 30, 1.3) + .035 * sand + .035 * ridges - .32 * p
    out.append((h, t, 2.0))
    # weathered concrete: sand and small stones exposed
    sh, mask, grey = stones(30000, .16)
    p = np.maximum(pits(300, 6), .9 * pits(40, 14))
    h = .32 * sh + .08 * noise(4, 40, 1.4) + .04 * sand - .9 * p
    t = np.where(mask, .38 + .3 * grey, .43) + .06 * noise(2, 30, 1.3) - .3 * p
    out.append((blur(h, .6), blur(t, .6), 1.8))
    # trowelled: smooth, long float arcs, sparse pits
    arcs = noise(6, 40, 1., stretch=3.)
    p = pits(70, 5)
    h = .1 * arcs + .025 * sand + .1 * noise(3, 20, 1.5) - .8 * p
    t = .5 + .04 * noise(2, 20, 1.4) + .015 * arcs + .02 * sand - .28 * p
    out.append((h, t, 1.2))
    return out + macro()


def macro():
    """Two 8 m tiles (8 mm a texel) for what reads from a moving car: 6 old asphalt, 7 old concrete."""
    out = []
    # asphalt: long wandering cracks sealed with tar, finer branching cracks, binder-rich and stone-rich mottling, stains
    long = cracks(9, 2.2)
    seal = np.clip(blur(long, 3.5) * 5, 0, 1) * (noise(1, 6, 1.) > -.2)        # sealant poured over most long cracks
    fine = cracks(70, 1.6) * np.clip(noise(1, 5, 1.) * 1.5, 0, 1)               # alligator cracking, in some areas only
    mottle = noise(3, 60, 1.3)
    stain = np.clip(noise(2, 14, 1.4) - 1.1, 0, 1.5)
    t = .5 + .09 * mottle - .16 * stain - .3 * seal - .28 * np.maximum(long, fine) + .03 * noise(60, 500, .3)
    h = .12 * mottle - .5 * np.maximum(long, fine) * (1 - seal) + .25 * seal
    out.append((blur(h, .7), blur(t, .7), 2.5))
    # concrete: stains and run-off, a few long cracks, patches worn smooth and pale
    long = cracks(5, 1.6)
    stain = np.clip(noise(2, 20, 1.5) - .7, 0, 2)
    pale = np.clip(noise(2, 10, 1.2) - .9, 0, 1)
    t = .5 + .07 * noise(3, 80, 1.2) - .13 * stain + .08 * pale - .35 * long + .025 * noise(80, 500, .3)
    h = .1 * noise(3, 60, 1.3) - .6 * long
    out.append((blur(h, .7), blur(t, .7), 2.0))
    return out


def pack(height, tone, relief):
    # 1 mm texels: a height unit is taken as `relief` millimetres
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * .5 * relief
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * .5 * relief
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    tone = tone - tone.mean() + .5
    rgb = np.stack([n[..., 0] * .5 + .5, n[..., 1] * .5 + .5, tone], -1)
    return (np.clip(rgb, 0, 1) * 255 + .5).astype(np.uint8)


if __name__ == '__main__':
    made = layers()
    tiles = [pack(*layer) for layer in made]
    Image.fromarray(np.concatenate(tiles, 0)).save(sys.argv[1], optimize=True)
    # the heights themselves, for parallax and self-shadowing: 1 at the surface's high points, 0 at its deepest
    heights = []
    for h, _, _ in made:
        lo, hi = np.percentile(h, .2), np.percentile(h, 99.5)
        heights.append((np.clip((h - lo) / (hi - lo), 0, 1) * 255 + .5).astype(np.uint8))
    Image.fromarray(np.concatenate(heights, 0)).save(sys.argv[1].replace('.png', '-height.png'), optimize=True)
    for i, tile in enumerate(tiles):
        print(i, 'tone sd', round(float(tile[..., 2].std()), 1), 'slope sd', round(float(tile[..., 0].std()), 1))
