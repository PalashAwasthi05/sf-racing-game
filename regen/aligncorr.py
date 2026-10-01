"""Apply a view's LiDAR skyline alignment (align.py) to a reading measured on that view: image metres -> wall metres."""
import remap

def choose(f):
    """The correction a view's fit supports: full (shift, height, scale), height only, or none."""
    if not f: return None
    if (not f.get('atBound') and f['cost'] < .6 * f['cost0'] and f['cost'] < .8 and f['shape'] >= .8):
        return {'kind': 'full', 'dx': f['dx'], 'dz': f['dz'], 's': f['s'], 'uc': f['uc'], 'eye': f['eye']}
    if f.get('zOnlyCost', 9) < .7 * f['cost0'] and f['zOnlyCost'] < .8:
        return {'kind': 'height', 'dx': 0., 'dz': f['zOnlyDz'], 's': 1., 'uc': f['uc'], 'eye': f['eye']}
    return None

class Affine:
    def __init__(self, c, L):
        self.c, self.L = c, L
    def __call__(self, x, z):
        c = self.c
        return c['uc'] + (x - c['dx'] - c['uc']) * c['s'], c['eye'] + (z - c['dz'] - c['eye']) * c['s']

def correct(r, piece, alignment):
    c = choose((alignment or {}).get(str(r.get('use', 0))))
    if not c: return r, None
    out = remap.remap_reading(r, Affine(c, float(piece['lengthM'])))
    out.pop('registered', None); out['aligned'] = c
    return out, c
