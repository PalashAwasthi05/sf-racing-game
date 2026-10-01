"""Move a reading measured on a provider-pose rectification onto the wall as the registered camera sees it.

A point (x, z) of the rectified image is the pixel whose ray from the provider pose meets the wall there. The same
pixel's ray from the registered pose meets the wall elsewhere; that is where the feature really is. Every coordinate of
a reading is carried through this map (rectangles through their corners). A registration is used only when it clearly
improved the skyline fit; otherwise the reading stays as measured."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

def good(reg):
    if not reg or reg.get('failed'): return False
    b, a = reg['before'], reg['after']
    return a['medianAbsDeg'] < .8 * b['medianAbsDeg'] and a['medianAbsDeg'] < 1.6 and a['inlierShare'] > .45 and np.hypot(*reg['shiftM'][:2]) < 8

class Map:
    def __init__(self, piece, cam, reg, z0, eye_h=2.4):
        n = np.asarray(piece['normal'], float); n /= np.linalg.norm(n); self.n = n; self.t = np.array([-n[1], n[0]])
        L = float(piece['lengthM']); u0, u1 = piece['frameU']; self.m = self.n * float(piece['planeV']) + self.t * ((u0 + u1) / 2)
        self.L, self.z0 = L, z0
        self.C0 = np.array([cam['xy'][0], cam['xy'][1], z0 + eye_h]); self.R0 = np.asarray(cam['R'], float)
        self.C1 = np.array([reg['xy'][0], reg['xy'][1], reg['z']]); self.R1 = Rotation.from_rotvec(reg['rotvec']).as_matrix()

    def __call__(self, x, z):
        P = np.r_[self.m[:2] + self.t * (x - self.L / 2) + self.n[:2] * .05, self.z0 + z]
        d_cam = self.R0 @ (P - self.C0)                  # camera frame direction of that pixel
        w = self.R1.T @ d_cam                             # the same pixel from the registered pose, in the world
        n3 = np.r_[self.n, 0.]; plane = float(self.m @ self.n) + .05
        den = float(w @ n3)
        if abs(den) < 1e-6: return x, z
        s = (plane - float(self.C1 @ n3)) / den
        if s <= 0: return x, z
        Q = self.C1 + s * w
        return float((Q[:2] - self.m[:2]) @ self.t + self.L / 2), float(Q[2] - self.z0)

def remap_reading(r, M):
    r = copy.deepcopy(r)
    def rect(o):
        (a, b), (c, d) = M(o['x0'], o['z0']), M(o['x1'], o['z1']); o['x0'], o['x1'] = sorted((a, c)); o['z0'], o['z1'] = sorted((b, d))
    for key in ('materials', 'banks', 'openings', 'bays', 'canopies'):
        for o in r.get(key) or []:
            if key == 'canopies':
                (a, zz), (c, _) = M(o['x0'], o['zM']), M(o['x1'], o['zM']); o['x0'], o['x1'] = sorted((a, c)); o['zM'] = zz; continue
            zm, xm = (o['z0'] + o['z1']) / 2, (o['x0'] + o['x1']) / 2
            kx = abs(M(o['x1'], zm)[0] - M(o['x0'], zm)[0]) / max(o['x1'] - o['x0'], 1e-6)       # local scale along
            kz = abs(M(xm, o['z1'])[1] - M(xm, o['z0'])[1]) / max(o['z1'] - o['z0'], 1e-6)       # and up
            if key == 'banks':
                if o.get('xCentersM'): o['xCentersM'] = [M(x, zm)[0] for x in o['xCentersM']]
                if o.get('zCentersM'): o['zCentersM'] = [M(xm, z)[1] for z in o['zCentersM']]
                if o.get('openingWidthM'): o['openingWidthM'] *= kx
                if o.get('openingHeightM'): o['openingHeightM'] *= kz
            rect(o)
    for b in r.get('bands') or []: b['zM'] = M(.5 * M.L, b['zM'])[1]
    for p in r.get('piers') or []: p['xM'] = M(p['xM'], 3.)[0]
    if r.get('wallTopM'): r['wallTopM'] = M(.5 * M.L, r['wallTopM'])[1]
    if r.get('parapetTop'): r['parapetTop'] = [list(M(x, z)) for x, z in r['parapetTop']]
    r['registered'] = True
    return r
