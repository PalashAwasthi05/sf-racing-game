"""Facade readings measured on photographs cut from a provider camera pose, moved onto the wall where the registered
camera (regen/register_cameras.py) puts what they measured.

A side's rectified photograph (auto_read_frontages_v24.rectify) samples the panorama along rays from the camera to points
of the wall plane: a camera metres off its true place sampled each wall point's ray in the wrong direction, so the crop
shows the wall shifted along itself, scaled and slightly sheared, and the reading measured on it carries the same error
(a front read 3 m to the left of where it stands, a wall a third blank). For a plane the error is exact: a point (x, z)
of the old crop (metres along the frame from its left end, metres above the side's ground line) lies on the pixel ray
R_nominal (P - C_nominal); the registered camera sees that pixel along R_registered^T R_nominal (P - C_nominal) from
C_registered, and where that ray meets the wall is where the photographed thing stands. Positions are moved, sizes are
scaled by the map's local stretch; what the old crop never showed stays unread (facade_lint reports it as `partial`).
Usage: from remap_readings import WallMap, remap"""
import copy
import numpy as np

BOX = ('materials', 'banks', 'openings', 'bays', 'ornaments', 'rustication', 'timbering', 'parapets')   # x0, x1, z0, z1
RUN = ('bands', 'balconies', 'canopies', 'roofs')                         # x0, x1 at zM (a band may span the whole frame)
GROUND = ('planting', 'streetEdge', 'hedges')                            # x0, x1 at the foot of the wall


class WallMap:
    """(x, z) of a side's old crop -> (x, z) on the wall, for one photograph."""

    def __init__(self, piece, R_nom, C_nom, R_reg, C_reg):
        n = np.asarray(piece['normal'], float)[:2]; n /= np.linalg.norm(n); t = np.array([-n[1], n[0]])
        u0, u1 = piece['frameU']; self.L = float(piece['lengthM']); self.z0 = float(piece['groundZ'])
        self.mid = n * float(piece['planeV']) + t * (u0 + u1) / 2; self.n, self.t = n, t; self.v = float(piece['planeV']) + .05
        self.M = np.asarray(R_reg).T @ np.asarray(R_nom); self.Cn, self.Cr = np.asarray(C_nom, float), np.asarray(C_reg, float)

    def __call__(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        P = np.stack([self.mid[0] + self.t[0] * (x - self.L / 2) + self.n[0] * .05, self.mid[1] + self.t[1] * (x - self.L / 2) + self.n[1] * .05, self.z0 + z], -1)
        d = (P - self.Cn) @ self.M.T; n3 = np.array([self.n[0], self.n[1], 0.])
        s = (self.v - self.Cr @ n3) / np.where(np.abs(d @ n3) < 1e-9, 1e-9, d @ n3); X = self.Cr + s[..., None] * d
        return (X[..., :2] - self.mid) @ self.t + self.L / 2, X[..., 2] - self.z0

    def stretch(self, x, z):
        """The map's local scale along x and along z at (x, z)."""
        (xa, za), (xb, zb), (xc, zc) = self(x, z), self(x + .5, z), self(x, z + .5)
        return float(abs(xb - xa) / .5), float(abs(zc - za) / .5)


def _num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


def remap(reading, wm):
    """A copy of `reading` with every position on its wall moved by `wm` (a WallMap) and its sizes scaled."""
    r = copy.deepcopy(reading); L = wm.L
    def box(o):
        xs, zs = wm([o['x0'], o['x1'], o['x0'], o['x1']], [o['z0'], o['z0'], o['z1'], o['z1']])
        cx, cz = (o['x0'] + o['x1']) / 2, (o['z0'] + o['z1']) / 2; kx, kz = wm.stretch(cx, cz)
        if isinstance(o.get('xCentersM'), list): o['xCentersM'] = [float(q) for q in wm([c for c in o['xCentersM'] if _num(c)], cz)[0]]
        if isinstance(o.get('zCentersM'), list): o['zCentersM'] = [float(q) for q in wm(cx, [c for c in o['zCentersM'] if _num(c)])[1]]
        for k in ('openingWidthM', 'postPitchM'): o[k] = o[k] * kx if _num(o.get(k)) else o.get(k)
        if isinstance(o.get('rails'), list): o['rails'] = [float(q) for q in wm(cx, [c for c in o['rails'] if _num(c)])[1]]
        for k in ('openingHeightM', 'transomM', 'courseM'): o[k] = o[k] * kz if _num(o.get(k)) else o.get(k)
        o['x0'], o['x1'] = float(min(xs)), float(max(xs)); o['z0'], o['z1'] = float(min(zs)), float(max(zs))
    for sec in BOX:
        for o in r.get(sec) or []:
            if isinstance(o, dict) and all(_num(o.get(k)) for k in ('x0', 'x1', 'z0', 'z1')): box(o)
    for sec in RUN:
        for o in r.get(sec) or []:
            if not isinstance(o, dict) or not _num(o.get('zM')): continue
            x0, x1 = (o['x0'], o['x1']) if _num(o.get('x0')) and _num(o.get('x1')) else (0., L)
            xs, zs = wm([x0, x1], [o['zM'], o['zM']]); kx, kz = wm.stretch((x0 + x1) / 2, o['zM'])
            if _num(o.get('x0')) and _num(o.get('x1')): o['x0'], o['x1'] = float(min(xs)), float(max(xs))
            o['zM'] = float(np.mean(zs))
            for k in ('thicknessM', 'riseM'): o[k] = o[k] * kz if _num(o.get(k)) else o.get(k)
            if _num(o.get('bracketPitchM')): o['bracketPitchM'] = o['bracketPitchM'] * kx
    for sec in GROUND:
        for o in r.get(sec) or []:
            if not (isinstance(o, dict) and _num(o.get('x0')) and _num(o.get('x1'))): continue
            xs, _ = wm([o['x0'], o['x1']], [.5, .5]); o['x0'], o['x1'] = float(min(xs)), float(max(xs))
            if isinstance(o.get('piersM'), list): o['piersM'] = [float(q) for q in wm([c for c in o['piersM'] if _num(c)], .5)[0]]
    for o in r.get('porches') or []:
        if isinstance(o, dict) and _num(o.get('x0')) and _num(o.get('x1')):
            zc = float(o.get('topM') or 3.) - .5; xs, _ = wm([o['x0'], o['x1']], [zc, zc]); o['x0'], o['x1'] = float(min(xs)), float(max(xs))
            for k in ('floorM', 'topM'):
                if _num(o.get(k)): o[k] = float(wm((o['x0'] + o['x1']) / 2, o[k])[1])
    for o in r.get('fireEscapes') or []:
        if isinstance(o, dict) and _num(o.get('x0')) and _num(o.get('x1')):
            cz = float(np.mean([v for v in (o.get('levels') or [3.]) if _num(v)] or [3.]))
            xs, _ = wm([o['x0'], o['x1']], [cz, cz]); o['x0'], o['x1'] = float(min(xs)), float(max(xs))
            if isinstance(o.get('levels'), list): o['levels'] = [float(q) for q in wm((o['x0'] + o['x1']) / 2, [v for v in o['levels'] if _num(v)])[1]]
    for o in r.get('piers') or []:
        if isinstance(o, dict) and _num(o.get('xM')):
            zc = (o['z0'] + o['z1']) / 2 if _num(o.get('z0')) and _num(o.get('z1')) else 3.
            xm, _ = wm(o['xM'], zc); kx, _ = wm.stretch(o['xM'], zc); o['xM'] = float(xm)
            if _num(o.get('widthM')): o['widthM'] = o['widthM'] * kx
            if _num(o.get('z0')) and _num(o.get('z1')): _, zz = wm(o['xM'], [o['z0'], o['z1']]); o['z0'], o['z1'] = float(min(zz)), float(max(zz))
    if _num(r.get('wallTopM')): r['wallTopM'] = float(wm(L / 2, r['wallTopM'])[1])
    if isinstance(r.get('storeys'), list): r['storeys'] = [float(q) for q in wm(L / 2, [v for v in r['storeys'] if _num(v)])[1]]
    if isinstance(r.get('parapetTop'), list):
        pts = [q for q in r['parapetTop'] if isinstance(q, (list, tuple)) and len(q) == 2 and _num(q[0]) and _num(q[1])]
        if pts: xs, zs = wm([q[0] for q in pts], [q[1] for q in pts]); r['parapetTop'] = [[float(a), float(b)] for a, b in zip(xs, zs)]
    return r
