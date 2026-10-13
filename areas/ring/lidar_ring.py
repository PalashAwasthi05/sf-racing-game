"""The LiDAR returns round the park's ring of buildings, in the world's local frame, for the massing tools: x, y (metres
east/north of the world centre), z (metres above the world datum, as the generator uses), the return class, and hag
(height above the ground under the return, from a 1 m ground grid of class-2 returns, filled where buildings stand).
Usage (run from work/): lidar_ring.py [out.npz]   (default areas/ring/lidar-ring.npz; SFW_AREA = the area)"""
import json, os, sys
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import box
from shapely.ops import unary_union
WORK = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(WORK)); os.environ.setdefault('SFW_AREA', str(WORK / 'areas/pacific-heights'))
import frontage_crops_v15 as fc

out = Path(sys.argv[1] if len(sys.argv) > 1 else WORK / 'areas/ring/lidar-ring.npz')
world, frame = fc.load_world(); fps = fc.load_footprints(frame)
ring = json.load(open(WORK / 'areas/ring/ring-ids.json'))
area = unary_union([fps[b.split('_p')[0]] for b in ring if b.split('_p')[0] in fps])
x0, y0, x1, y1 = area.buffer(35).bounds
pts = np.load(Path(os.environ['SFW_AREA']) / 'data/points.npz')['points']
t = Transformer.from_crs(3857, 32610, always_xy=True); X, Y = t.transform(pts[:, 0], pts[:, 1]); X = X - frame.cx; Y = Y - frame.cy
keep = (X > x0) & (X < x1) & (Y > y0) & (Y < y1); cls = pts[:, 3]
# the generator's datum: the median ground return within 40 m of the centre
g = cls == 2; datum = float(np.median(pts[g & (np.hypot(X, Y) < 40), 2]))
X, Y, Z, C = X[keep], Y[keep], pts[keep, 2] - datum, cls[keep]
keep = (C != 7) & (C != 18); X, Y, Z, C = X[keep], Y[keep], Z[keep], C[keep]      # noise classes out
# a 1 m ground grid: the lowest class-2 return per cell, gaps (under buildings) filled from the nearest measured cell
nx, ny = int(np.ceil(x1 - x0)) + 1, int(np.ceil(y1 - y0)) + 1; G = np.full((ny, nx), np.inf)
gi = C == 2; ix = ((X[gi] - x0)).astype(int); iy = ((Y[gi] - y0)).astype(int); np.minimum.at(G, (iy, ix), Z[gi])
known = np.isfinite(G); _, (ii, jj) = distance_transform_edt(~known, return_indices=True); G = G[ii, jj]; G = gaussian_filter(G, 1.)
hag = Z - G[((Y - y0)).astype(int).clip(0, ny - 1), ((X - x0)).astype(int).clip(0, nx - 1)]
np.savez_compressed(out, x=X.astype(np.float32), y=Y.astype(np.float32), z=Z.astype(np.float32), cls=C.astype(np.uint8), hag=hag.astype(np.float32),
                    ground=G.astype(np.float32), origin=np.array([x0, y0]), datum=np.array([datum]))
print(json.dumps({'out': str(out), 'points': int(len(X)), 'bbox': [round(v, 1) for v in (x0, y0, x1, y1)], 'datum': round(datum, 3)}))
