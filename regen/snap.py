"""Snap a reading's openings and bands onto the photograph's edges.

Models count and classify well but place outlines a few decimetres off. Each bank, single opening and band is moved
within a small window (and its opening size and spacing adjusted a little) to where its outlines lie on the strongest
edges of the rectified photograph; a move is kept only if it clearly improves that fit. Pure image geometry."""
import copy, json, pathlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import specs


def edges(path, pad):
    g = np.asarray(Image.open(path).convert('L'), float)[pad:-pad, pad:-pad]
    g = ndi.gaussian_filter(g, 1.2)
    return np.abs(ndi.sobel(g, 1)), np.abs(ndi.sobel(g, 0))      # |d/dx| (vertical edges), |d/dz| (horizontal edges)


class Fit:
    def __init__(self, gx, gy, ppm, H):
        self.gx, self.gy, self.ppm, self.H = gx, gy, ppm, H; self.h, self.w = gx.shape
        self.norm = float(np.percentile(np.r_[gx.ravel(), gy.ravel()], 90)) + 1e-6

    def _line(self, img, x0, z0, x1, z1, n=24):
        xs = np.linspace(x0, x1, n) * self.ppm; ys = (self.H - np.linspace(z0, z1, n)) * self.ppm
        ok = (xs >= 0) & (xs < self.w - 1) & (ys >= 0) & (ys < self.h - 1)
        if ok.sum() < n * .5: return None
        return float(ndi.map_coordinates(img, [ys[ok], xs[ok]], order=1).mean())

    def rects(self, rs):
        s, k = 0., 0
        L, H = self.w / self.ppm, self.H
        for x0, z0, x1, z1 in rs:
            if x0 < -.05 or x1 > L + .05 or z0 < -.05 or z1 > H - .05: continue
            for v in (self._line(self.gx, x0, z0, x0, z1), self._line(self.gx, x1, z0, x1, z1),
                      self._line(self.gy, x0, z0, x1, z0), self._line(self.gy, x0, z1, x1, z1)):
                if v is not None: s += v; k += 1
        return s / k / self.norm if k else 0.

    def band(self, z, L):
        return (self._line(self.gy, 0, z, L, z, 64) or 0.) / self.norm


def _search(score, p, steps):
    """Coordinate search from parameters p over the given (key, values) steps; best of each in turn, twice."""
    best, bs = dict(p), score(p)
    for _ in range(2):
        for key, vals in steps:
            for v in vals:
                q = dict(best); q[key] = v
                s = score(q)
                if s > bs: best, bs = q, s
    return best, bs


def snap_reading(r, piece, img_dir, gain=1.05):
    v = piece['views'][r.get('use', 0)]
    gx, gy = edges(pathlib.Path(img_dir) / v['rect'], v['rectPad'])
    F = Fit(gx, gy, v['ppm'], float(piece['heightM'])); L = float(piece['lengthM'])
    r = copy.deepcopy(r); report = []
    for k, b in enumerate(r.get('banks') or []):
        nx, ny = max(1, int(b.get('columns', 1))), max(1, int(b.get('rows', 1)))
        dx0, dz0 = (b['x1'] - b['x0']) / nx, (b['z1'] - b['z0']) / ny
        w0 = min(dx0 * .98, b.get('openingWidthM') or dx0 * .7); h0 = min(dz0 * .98, b.get('openingHeightM') or dz0 * .7)
        base = {'sx': 0., 'sz': 0., 'px': 1., 'pz': 1., 'w': w0, 'h': h0}
        def cells(p):
            bb = dict(b, x0=b['x0'] + p['sx'], z0=b['z0'] + p['sz'])
            bb['x1'] = bb['x0'] + (b['x1'] - b['x0']) * p['px']; bb['z1'] = bb['z0'] + (b['z1'] - b['z0']) * p['pz']
            bb['openingWidthM'], bb['openingHeightM'] = p['w'], p['h']
            return bb, specs.bank_cells(bb)
        score = lambda p: F.rects(cells(p)[1])
        s0 = score(base)
        best, bs = _search(score, base, [('sx', np.arange(-.9, .91, .1)), ('sz', np.arange(-.7, .71, .1)),
                                          ('px', np.arange(.94, 1.061, .01)), ('pz', np.arange(.94, 1.061, .01)),
                                          ('w', w0 * np.arange(.75, 1.26, .05)), ('h', h0 * np.arange(.75, 1.26, .05))])
        bb = cells(best)[0] if bs > s0 * gain else dict(b, openingWidthM=w0, openingHeightM=h0)
        # then each column and each row on its own (real windows are often unevenly spaced)
        dxb, dzb = (bb['x1'] - bb['x0']) / nx, (bb['z1'] - bb['z0']) / ny
        xc = [bb['x0'] + (i + .5) * dxb for i in range(nx)]; zc = [bb['z0'] + (j + .5) * dzb for j in range(ny)]
        W_, H_ = bb['openingWidthM'], bb['openingHeightM']; omit = {tuple(o) for o in bb.get('omit', [])}
        def col_rects(i, cx): return [(cx - W_ / 2, z - H_ / 2, cx + W_ / 2, z + H_ / 2) for j, z in enumerate(zc) if (i, j) not in omit]
        def row_rects(j, cz): return [(x - W_ / 2, cz - H_ / 2, x + W_ / 2, cz + H_ / 2) for i, x in enumerate(xc) if (i, j) not in omit]
        lim = min(.45, .3 * dxb)
        for i in range(nx):
            cand = xc[i] + np.arange(-lim, lim + 1e-6, .05); sc = [F.rects(col_rects(i, c)) for c in cand]; j = int(np.argmax(sc))
            if sc[j] > F.rects(col_rects(i, xc[i])) * gain: xc[i] = float(cand[j])
        limz = min(.35, .3 * dzb)
        for j in range(ny):
            cand = zc[j] + np.arange(-limz, limz + 1e-6, .05); sc = [F.rects(row_rects(j, c)) for c in cand]; i = int(np.argmax(sc))
            if sc[i] > F.rects(row_rects(j, zc[j])) * gain: zc[j] = float(cand[i])
        bb = {**bb, 'x0': round(bb['x0'], 3), 'x1': round(bb['x1'], 3), 'z0': round(bb['z0'], 3), 'z1': round(bb['z1'], 3),
              'openingWidthM': round(W_, 3), 'openingHeightM': round(H_, 3), 'xCentersM': [round(x, 3) for x in xc], 'zCentersM': [round(z, 3) for z in zc]}
        r['banks'][k] = bb; s1 = F.rects(specs.bank_cells(bb))
        report.append(('B%d' % k, round(s0, 3), round(s1, 3), s1 > s0 * gain))
    for k, o in enumerate(r.get('openings') or []):
        base = {'sx': 0., 'sz': 0., 'w': o['x1'] - o['x0'], 'h': o['z1'] - o['z0']}
        def rect(p): return (o['x0'] + p['sx'], o['z0'] + p['sz'], o['x0'] + p['sx'] + p['w'], o['z0'] + p['sz'] + p['h'])
        score = lambda p: F.rects([rect(p)])
        s0 = score(base)
        best, bs = _search(score, base, [('sx', np.arange(-.7, .71, .1)), ('sz', np.arange(-.5, .51, .1)),
                                          ('w', base['w'] * np.arange(.8, 1.21, .05)), ('h', base['h'] * np.arange(.8, 1.21, .05))])
        if bs > s0 * gain:
            x0, z0, x1, z1 = rect(best); r['openings'][k] = {**o, 'x0': round(max(0, x0), 3), 'z0': round(max(0, z0), 3), 'x1': round(min(L, x1), 3), 'z1': round(z1, 3)}
        report.append(('O%d' % k, round(s0, 3), round(max(bs, s0), 3), bs > s0 * gain))
    for k, b in enumerate(r.get('bands') or []):
        zs = b['zM'] + np.arange(-.5, .51, .05); sc = [F.band(z, L) for z in zs]; s0 = F.band(b['zM'], L); i = int(np.argmax(sc))
        if sc[i] > s0 * gain: r['bands'][k] = {**b, 'zM': round(float(zs[i]), 3)}
        report.append(('L%d' % k, round(s0, 3), round(max(sc[i], s0), 3), sc[i] > s0 * gain))
    return r, report
