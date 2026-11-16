"""Generate the tiling wall finishes the viewer projects on building walls (app/surface-finish.ts).

Five 1024 x 1024 layers stacked top to bottom in one RGB image, each a 2.4 m square (2.3 mm a texel), with relief sized
to read from the street (features of 2 to 15 cm), not only from arm's length:
  0 stucco, lace: flat-topped islands of a second coat standing a few millimetres proud of a sandy base
  1 stucco, floated: sand grain, the float's short sweeps, the wall's slow unevenness
  2 cast stone / concrete: pitted, blotched, a little sandy
  3 painted: roller stipple, sags running down, hairline crazing
  4 masonry face: grit, pits and scuffs for the face of brick and stone that already carries its own pattern
R, G: the surface normal's x, y (0.5 = flat); B: tone about 0.5, darker in the recesses (so depth shows in shade too).
Everything wraps. Run: uv run --python 3.11 --with numpy --with scipy --with pillow python generate_walls.py <out.png>
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

N = 1024
MM = 2400 / N
rng = np.random.default_rng(431)
yy, xx = np.mgrid[0:N, 0:N]
grid = np.stack([xx.ravel() + .5, yy.ravel() + .5], 1)
fx = np.fft.fftfreq(N)[None, :] * N
fy = np.fft.fftfreq(N)[:, None] * N
radial = np.hypot(fx, fy)


def noise(lo, hi, slope=1., sx=1., sy=1.):
    """Band-limited noise between `lo` and `hi` cycles a tile, 1/f^slope; sx > 1 draws it out along x, sy along y."""
    r = np.hypot(fx * sx, fy * sy)
    amp = np.where((r >= lo) & (r <= hi), 1. / np.maximum(r, 1.) ** slope, 0.)
    out = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * amp).real
    return out / out.std()


def blur(a, sigma):
    return np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (np.pi * sigma * radial / N) ** 2)).real


def cells(count):
    pts = rng.random((count, 2)) * N
    d, i = cKDTree(pts, boxsize=N).query(grid, k=2)
    return d[:, 0].reshape(N, N), d[:, 1].reshape(N, N), i[:, 0].reshape(N, N), rng.random(count)


def pits(count, radius):
    d1, _, idx, val = cells(count)
    return np.clip(1. - d1 / (radius * (.4 + 1.2 * val[idx])), 0, 1) ** .7


def smooth(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def cavity(h, sigma):
    """Darker where the surface is below its surroundings."""
    return h - blur(h, sigma)


def layers():
    out = []
    # 0 lace stucco: islands 4 to 14 cm across, 5 mm proud, their tops trowelled flat; sandy base between them
    field = noise(9, 34, .6) + .35 * noise(34, 90, .4)
    top = smooth(field, -.05, .3)
    grain = noise(140, 512, .2)
    h = 5. * top + .5 * grain * (1 - .6 * top) + 1.2 * noise(2, 7, 1.)
    tone = .5 + .022 * cavity(h, 10) + .02 * grain * (1 - top) - .012 * (1 - top) + .03 * noise(2, 10, 1.)
    out.append((h, tone))
    # 1 floated stucco: sand grain, short float sweeps in two directions, slow unevenness of the wall
    grain = noise(110, 512, .25)
    sweeps = noise(14, 90, .5, sx=5.) * smooth(noise(3, 9), -.3, .5) + noise(14, 90, .5, sy=5.) * smooth(noise(3, 9), -.3, .5)
    h = .7 * grain + .65 * sweeps + 3.2 * noise(2, 9, 1.) + .9 * noise(9, 30, .8)
    tone = .5 + .035 * cavity(h, 9) + .025 * grain + .035 * noise(2, 8, 1.)
    out.append((h, tone))
    # 2 cast stone / concrete: bug holes, sandy face, blotches
    holes = pits(520, 7) + .7 * pits(1600, 3.5)
    grain = noise(90, 512, .3)
    h = -3.2 * holes + .6 * grain + 2.2 * noise(2, 8, 1.) + .7 * noise(8, 40, .8)
    tone = .5 - .07 * smooth(holes, .15, .8) + .025 * grain + .035 * noise(2, 7, 1.) + .02 * noise(7, 30, .8)
    out.append((h, tone))
    # 3 painted: roller stipple, sags running down the wall, hairline crazing
    stipple = noise(120, 420, .2)
    sags = noise(10, 70, .6, sy=7.) * smooth(noise(2, 6), 0, .8)
    d1, d2, _, _ = cells(70)
    craze = np.clip(1. - ((d2 - d1) + 2. * noise(6, 60, 1.2)) / 1.6, 0, 1) * smooth(noise(2, 5), .2, .9)
    h = .55 * stipple + 1.4 * sags - 1.8 * craze + 2.6 * noise(2, 8, 1.)
    tone = .5 + .03 * cavity(h, 8) - .12 * craze + .025 * noise(2, 9, 1.)
    out.append((h, tone))
    # 4 masonry face: grit, small pits, scuffs along the course
    grit = noise(100, 512, .15)
    holes = pits(2600, 3.)
    scuff = noise(20, 160, .5, sx=6.) * smooth(noise(3, 10), .1, .8)
    h = 1.1 * grit - 2.2 * holes + .8 * scuff + 1.2 * noise(4, 20, .9)
    tone = .5 + .04 * grit - .08 * smooth(holes, .2, .8) + .03 * scuff + .03 * noise(3, 14, 1.)
    out.append((h, tone))
    return out


def pack(height, tone):
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * .5 / MM
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * .5 / MM
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    tone = tone - tone.mean() + .5
    rgb = np.stack([n[..., 0] * .5 + .5, n[..., 1] * .5 + .5, tone], -1)
    return (np.clip(rgb, 0, 1) * 255 + .5).astype(np.uint8)


if __name__ == '__main__':
    tiles = [pack(*layer) for layer in layers()]
    Image.fromarray(np.concatenate(tiles, 0)).save(sys.argv[1], optimize=True)
    for i, tile in enumerate(tiles):
        print(i, 'tone sd', round(float(tile[..., 2].std()), 1), 'slope sd', round(float(tile[..., 0].std()), 1))
