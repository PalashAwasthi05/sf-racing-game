#!/usr/bin/env python
"""Frontage crops from cached street panoramas (v15).

Cheap check of whether the cached equirectangular panoramas can supply a clean
frontage view for buildings whose facades are still unverified priors.  No
pixel-accurate registration is attempted: provider poses are used as they are
(metre-level), which is enough to point a virtual perspective camera at a wall.

Inputs (read only)
  king-street/public/world.json                  building list, local frame origin
  detailed-v14i/king-street-evidence.json        facadeWalls (midpoint, normal, length, height)
  imagery-street-v12/source-index-registered.json  camera poses (OpenSfM convention)
  imagery-street-v12/images/<id>.jpg             equirectangular panoramas
  data/osm.xml, data/buildings.json              footprints used as 2D occluders
  data/trees.json                                street trees (occlusion hint only)

Outputs
  facade-priority-v15/crops/<buildingId>__<wallIndex>__<imageId>.jpg
  facade-priority-v15/crops/manifest.json

Modes
  default            select cameras per wall, render crops, write manifest
  --probe IMAGE_ID   render an orientation probe for one panorama: a downscaled
                     panorama with every nearby wall drawn where the pose predicts
                     it, plus a perspective crop toward the nearest visible wall

No location-specific facts: buildings are addressed only by the ids carried in
the data files, and every rule below is geometric.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

WORK = pathlib.Path(__file__).resolve().parent
WORLD_PATH = WORK / 'king-street/public/world.json'
EVIDENCE_PATH = WORK / 'detailed-v14i/king-street-evidence.json'
SOURCE_INDEX_PATH = WORK / 'imagery-street-v12/source-index-registered.json'
IMAGE_DIR = WORK / 'imagery-street-v12/images'
OSM_PATH = WORK / 'data/osm.xml'
BUILDINGS_PATH = WORK / 'data/buildings.json'
TREES_PATH = WORK / 'data/trees.json'
OUT_DIR = WORK / 'facade-priority-v15/crops'
PROBE_DIR = WORK / 'facade-priority-v15/orientation-probe'

# Selection rules (general, geometric).
MAX_DISTANCE_M = 45.0          # cameras farther than this are not considered
FALLBACK_DISTANCE_M = 80.0     # used only when a building has no camera within MAX_DISTANCE_M
PREFERRED_DISTANCE_M = (12.0, 35.0)
MIN_INCIDENCE = 0.5            # cos of angle between wall normal and camera direction
MIN_VISIBLE_FRACTION = 0.5     # share of wall sample points with an unobstructed sight line
WALL_SAMPLES = 9               # sight-line samples along a wall
SIGHT_OFFSET_M = 0.3           # sight lines end this far in front of the wall
EYE_HEIGHT_M = 2.4             # camera height above the wall's ground line (fixed; GPS altitude is unreliable)
FOV_MARGIN = 1.15
MAX_HFOV_DEG = 110.0
MIN_HFOV_DEG = 40.0
MAX_VFOV_DEG = 100.0
CROP_WIDTH_PX = 1400
MAX_CROP_HEIGHT_PX = 1600
JPEG_QUALITY = 82
MIN_SECOND_VIEW_SEPARATION_M = 5.0

RANKING_PATH = WORK / 'facade-priority-v15/ranking.json'


def default_targets():
    """Buildings that deserve detail and lack it: prior-only buildings inside the ranking's 85% driver-visibility
    set (rank_facade_priority_v15.py). Read from the ranking output, never listed in code."""
    summary = json.load(open(RANKING_PATH))['summary']
    return list(summary['priorOnlyIn85pct'])


# ----------------------------------------------------------------------------- frames


class LocalFrame:
    """World local frame: UTM zone 10N metres relative to world.json 'center' (x east, y north)."""

    def __init__(self, center_lon: float, center_lat: float):
        self.ll = Transformer.from_crs(4326, 32610, always_xy=True)
        self.cx, self.cy = self.ll.transform(center_lon, center_lat)

    def local(self, lon, lat, z=None):
        a, b = self.ll.transform(lon, lat)
        return np.asarray(a) - self.cx, np.asarray(b) - self.cy


def load_world():
    world = json.load(open(WORLD_PATH))
    frame = LocalFrame(world['center']['lon'], world['center']['lat'])
    return world, frame


def load_footprints(frame: LocalFrame) -> dict[str, Polygon]:
    geoms = {}
    osm = ET.parse(OSM_PATH).getroot()
    nodes = {n.attrib['id']: (float(n.attrib['lon']), float(n.attrib['lat'])) for n in osm.findall('node')}
    for way in osm.findall('way'):
        tags = {a.attrib['k']: a.attrib['v'] for a in way.findall('tag')}
        coords = [nodes[a.attrib['ref']] for a in way.findall('nd') if a.attrib['ref'] in nodes]
        if tags.get('building') and len(coords) > 3:
            x, y = frame.local(*zip(*coords))
            poly = Polygon(np.c_[x, y]).buffer(0)
            if not poly.is_empty:
                geoms['osm_' + way.attrib['id']] = poly
    for b in json.load(open(BUILDINGS_PATH)):
        poly = transform(frame.local, shape(b['shape']))
        if not poly.is_empty:
            geoms['sf_' + b['sf16_bldgid'].replace('.', '_')] = poly
    return geoms


def load_trees(frame: LocalFrame) -> np.ndarray:
    try:
        rows = json.load(open(TREES_PATH))
    except FileNotFoundError:
        return np.zeros((0, 2))
    pts = []
    for r in rows:
        try:
            x, y = frame.local(float(r['longitude']), float(r['latitude']))
            pts.append((float(x), float(y)))
        except (KeyError, ValueError, TypeError):
            continue
    return np.array(pts) if pts else np.zeros((0, 2))


def load_cameras(frame: LocalFrame, rotation_transpose: bool = False) -> dict[str, dict]:
    """Camera records in the local frame.

    OpenSfM convention: rotationAngleAxis is the axis-angle of R with
    camera_point = R @ world_point + t, world = local topocentric (x east, y north,
    z up), camera = (x right, y down, z forward).
    """
    src = json.load(open(SOURCE_INDEX_PATH))
    cams = {}
    for key, rec in src.items():
        cam = rec['camera']
        image_id = key.split(':', 1)[1]
        lon, lat = cam['locationWgs84']
        x, y = frame.local(lon, lat)
        R = Rotation.from_rotvec(np.asarray(cam['rotationAngleAxis'], float)).as_matrix()
        if rotation_transpose:
            R = R.T
        path = IMAGE_DIR / f'{image_id}.jpg'
        cams[image_id] = {
            'id': image_id,
            'xy': np.array([float(x), float(y)]),
            'altitudeM': cam.get('altitudeM'),
            'R': R,
            'captureDate': rec.get('captureDate'),
            'registrationGrade': cam.get('registrationGrade'),
            'clusterId': cam.get('clusterId'),
            'batch': rec.get('batch'),                 # fetched later (fetch_panoramas_v32.py); None for the original set
            'path': path,
            'exists': path.exists(),
        }
    return cams


# ----------------------------------------------------------------------------- projection


def world_dirs_to_pano_uv(dirs_world: np.ndarray, R: np.ndarray, mirror_u: bool = False):
    """Map unit (or any) world direction vectors (..., 3) to equirectangular (u, v) in [0,1)."""
    d = dirs_world @ R.T  # camera coordinates
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(x, z)                       # bearing around the camera y axis, 0 at +z
    lat = np.arctan2(-y, np.hypot(x, z))         # elevation, positive up (camera y points down)
    u = lon / (2 * np.pi) + 0.5
    v = 0.5 - lat / np.pi
    if mirror_u:
        u = 1.0 - u
    return u % 1.0, np.clip(v, 0.0, 1.0)


def view_basis(forward: np.ndarray):
    """Right-handed camera basis (right, down, forward) for a world forward direction, world z up."""
    f = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return right, down, f


def render_perspective(pano: np.ndarray, R: np.ndarray, forward: np.ndarray, hfov_deg: float, vfov_deg: float,
                       width: int = CROP_WIDTH_PX, max_height: int = MAX_CROP_HEIGHT_PX, mirror_u: bool = False):
    """Inverse-map a pinhole view (looking along `forward`) into the panorama, bilinear sampling."""
    right, down, f = view_basis(forward)
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    height = int(round(2.0 * fx * math.tan(math.radians(vfov_deg) / 2.0)))
    if height > max_height:
        height = max_height
        vfov_deg = math.degrees(2.0 * math.atan((height / 2.0) / fx))
    xs = (np.arange(width) + 0.5 - width / 2.0) / fx
    ys = (np.arange(height) + 0.5 - height / 2.0) / fx
    dirs = (right[None, None, :] * xs[None, :, None] + down[None, None, :] * ys[:, None, None] + f[None, None, :])
    u, v = world_dirs_to_pano_uv(dirs, R, mirror_u)
    ph, pw = pano.shape[:2]
    px = (u * pw - 0.5) % pw
    py = np.clip(v * ph - 0.5, 0, ph - 1)
    out = np.empty((height, width, 3), dtype=np.uint8)
    coords = np.stack([py, px])
    for c in range(3):
        # wrap horizontally so the seam interpolates correctly
        out[..., c] = map_coordinates(pano[..., c], coords, order=1, mode='grid-wrap').astype(np.uint8)
    return out, vfov_deg


_PANO_CACHE: dict[str, np.ndarray] = {}


def load_pano(path: pathlib.Path) -> np.ndarray:
    key = str(path)
    if key not in _PANO_CACHE:
        if len(_PANO_CACHE) >= 3:
            _PANO_CACHE.pop(next(iter(_PANO_CACHE)))
        with Image.open(path) as im:
            _PANO_CACHE[key] = np.asarray(im.convert('RGB'))
    return _PANO_CACHE[key]


# ----------------------------------------------------------------------------- geometry


def wall_geometry(wall: dict):
    m = np.asarray(wall['midpoint'], float)
    n = np.asarray(wall['normal'], float)
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])  # along the wall
    half = wall['lengthM'] / 2.0
    return m, n, t, m - t * half, m + t * half


class Occluders:
    """2D footprint occlusion with an STRtree; a wall's own building footprint is excluded per query."""

    def __init__(self, footprints: dict[str, Polygon]):
        self.ids = list(footprints)
        self.polys = [footprints[i] for i in self.ids]
        self.tree = STRtree(self.polys)

    def own_indices(self, inside_point: Point) -> set[int]:
        return {int(i) for i in self.tree.query(inside_point, predicate='intersects')}

    def blocked(self, line: LineString, exclude: set[int]) -> bool:
        for i in self.tree.query(line, predicate='intersects'):
            if int(i) not in exclude:
                return True
        return False


def visible_fraction(cam_xy: np.ndarray, wall: dict, occ: Occluders, samples: int = WALL_SAMPLES):
    m, n, t, a, b = wall_geometry(wall)
    exclude = occ.own_indices(Point(*(m - n * 0.5)))
    exclude |= occ.own_indices(Point(*(m - n * 1.5)))
    hits = 0
    mid_ok = False
    for k in range(samples):
        s = (k + 0.5) / samples
        p = a + (b - a) * s
        ok = not occ.blocked(LineString([cam_xy, p + n * SIGHT_OFFSET_M]), exclude)
        hits += ok
        if k == samples // 2:
            mid_ok = ok
    return hits / samples, mid_ok


def distance_preference(d: float) -> float:
    lo, hi = PREFERRED_DISTANCE_M
    if d < lo:
        return max(0.0, d / lo)
    if d <= hi:
        return 1.0
    return max(0.0, 1.0 - (d - hi) / 25.0)


def trees_near_sight_line(trees: np.ndarray, cam_xy: np.ndarray, wall: dict, radius: float = 2.5) -> int:
    if len(trees) == 0:
        return 0
    m, n, t, a, b = wall_geometry(wall)
    line = LineString([cam_xy, m + n * SIGHT_OFFSET_M])
    # cheap prefilter by bounding box
    lo = np.minimum(cam_xy, m) - radius
    hi = np.maximum(cam_xy, m) + radius
    box = trees[(trees[:, 0] >= lo[0]) & (trees[:, 0] <= hi[0]) & (trees[:, 1] >= lo[1]) & (trees[:, 1] <= hi[1])]
    return int(sum(line.distance(Point(*p)) <= radius for p in box))


def view_parameters(cam_xy: np.ndarray, wall: dict, ground_z: float):
    """Yaw to the wall's angular centre, pitch to the angular centre of its vertical extent, FOVs with margin."""
    m, n, t, a, b = wall_geometry(wall)
    v_a = a - cam_xy
    v_b = b - cam_xy
    v_m = m - cam_xy
    yaw_m = math.atan2(v_m[0], v_m[1])  # compass bearing (0 north, clockwise) of the midpoint direction
    # signed angles of the endpoints relative to the midpoint direction
    def rel(v):
        ang = math.atan2(v[0], v[1]) - yaw_m
        return (ang + math.pi) % (2 * math.pi) - math.pi
    ra, rb = rel(v_a), rel(v_b)
    yaw = yaw_m + (ra + rb) / 2.0
    span_h = abs(rb - ra)
    hfov = math.degrees(span_h) * FOV_MARGIN
    hfov_capped = hfov > MAX_HFOV_DEG
    hfov = min(MAX_HFOV_DEG, max(MIN_HFOV_DEG, hfov))
    # vertical: horizontal distance to the wall centre line along the view direction
    d_mid = float(np.linalg.norm(v_m))
    cam_z = ground_z + EYE_HEIGHT_M
    ang_bottom = math.atan2(ground_z - cam_z, d_mid)
    ang_top = math.atan2(ground_z + wall['heightM'] - cam_z, d_mid)
    span_v = (ang_top - ang_bottom)
    vfov = math.degrees(span_v) * FOV_MARGIN
    vfov_capped = vfov > MAX_VFOV_DEG
    vfov = min(MAX_VFOV_DEG, max(30.0, vfov))
    if vfov_capped:
        # keep the ground floor in frame; the top is cut off
        pitch = ang_bottom - math.radians(vfov) * (FOV_MARGIN - 1.0) / 2.0 + math.radians(vfov) / 2.0
    else:
        pitch = (ang_bottom + ang_top) / 2.0
    forward = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    return {
        'forward': forward,
        'yawDeg': round((math.degrees(yaw) + 360.0) % 360.0, 2),
        'pitchDeg': round(math.degrees(pitch), 2),
        'hfovDeg': round(hfov, 2),
        'vfovDeg': round(vfov, 2),
        'wallSpanDeg': round(math.degrees(span_h), 2),
        'hfovCapped': bool(hfov_capped),
        'vfovCapped': bool(vfov_capped),
    }


# ----------------------------------------------------------------------------- candidates


def wall_candidates(wall: dict, cams: dict, occ: Occluders, trees: np.ndarray, max_distance: float):
    m, n, t, a, b = wall_geometry(wall)
    out = []
    for cam in cams.values():
        if not cam['exists']:
            continue
        v = cam['xy'] - m
        d = float(np.linalg.norm(v))
        if d < 2.0 or d > max_distance:
            continue
        inc = float(v @ n) / d
        if inc < MIN_INCIDENCE:
            continue
        vf, mid_ok = visible_fraction(cam['xy'], wall, occ)
        if not mid_ok or vf < MIN_VISIBLE_FRACTION:
            continue
        span = abs(math.atan2(*(a - cam['xy'])[::-1]) - math.atan2(*(b - cam['xy'])[::-1]))
        span = min(span, 2 * math.pi - span)
        fov_factor = min(1.0, math.radians(MAX_HFOV_DEG) / max(1e-6, span * FOV_MARGIN))
        score = distance_preference(d) * inc * vf * fov_factor
        out.append({
            'image': cam['id'],
            'distanceM': round(d, 2),
            'incidence': round(inc, 3),
            'visibleFraction': round(vf, 3),
            'treesNearSightLine': trees_near_sight_line(trees, cam['xy'], wall),
            'score': round(score, 4),
            'cameraLocal': [round(float(cam['xy'][0]), 2), round(float(cam['xy'][1]), 2)],
        })
    out.sort(key=lambda c: -c['score'])
    return out


def select_crops(buildings: dict[str, dict], targets: list[str], walls_by_building: dict[str, list[dict]],
                 cams: dict, occ: Occluders, trees: np.ndarray, budget: int):
    """Per wall best 1-2 cameras; every wall with a candidate gets one crop before any gets a second."""
    per_wall = []
    for bid in targets:
        for wi, wall in enumerate(walls_by_building.get(bid, [])):
            cands = wall_candidates(wall, cams, occ, trees, MAX_DISTANCE_M)
            per_wall.append({'building': bid, 'wallIndex': wi, 'wall': wall, 'candidates': cands, 'fallback': False})
    # fallback: a building with no camera within MAX_DISTANCE_M gets its single best distant view
    for bid in targets:
        mine = [w for w in per_wall if w['building'] == bid]
        if mine and not any(w['candidates'] for w in mine):
            best = None
            for w in mine:
                cands = wall_candidates(w['wall'], cams, occ, trees, FALLBACK_DISTANCE_M)
                if cands and (best is None or cands[0]['score'] > best[1]['score']):
                    best = (w, cands[0])
            if best:
                best[0]['candidates'] = [best[1]]
                best[0]['fallback'] = True
    chosen = []
    order = sorted([w for w in per_wall if w['candidates']], key=lambda w: -w['wall']['lengthM'])
    for w in order:  # first pass: best view per wall
        if len(chosen) >= budget:
            break
        chosen.append((w, w['candidates'][0], 1))
    for w in order:  # second pass: a second, spatially distinct view for the longer walls
        if len(chosen) >= budget:
            break
        if w['wall']['lengthM'] < 8.0 or w['fallback']:
            continue
        first = w['candidates'][0]
        for c in w['candidates'][1:]:
            sep = math.dist(c['cameraLocal'], first['cameraLocal'])
            if sep >= MIN_SECOND_VIEW_SEPARATION_M and c['score'] >= 0.35 * first['score']:
                chosen.append((w, c, 2))
                break
    return per_wall, chosen


# ----------------------------------------------------------------------------- probe


def run_probe(image_id: str, cams: dict, walls: list[dict], buildings: dict, occ: Occluders,
              mirror_u: bool, rotation_transpose: bool):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cam = cams[image_id]
    pano = load_pano(cam['path'])
    ph, pw = pano.shape[:2]
    scale = 1920 / pw
    small = Image.fromarray(pano).resize((1920, int(round(ph * scale))), Image.BILINEAR)
    draw = ImageDraw.Draw(small)
    labels = []
    nearest = None
    for wall in walls:
        m, n, t, a, b = wall_geometry(wall)
        v = cam['xy'] - m
        d = float(np.linalg.norm(v))
        if d > 60:
            continue
        inc = float(v @ n) / d
        if inc <= 0.05:
            continue
        vf, mid_ok = visible_fraction(cam['xy'], wall, occ)
        if not mid_ok:
            continue
        gz = buildings[wall['buildingId']]['center'][2]
        cz = gz + EYE_HEIGHT_M
        h = wall['heightM']
        corners = np.array([[a[0], a[1], gz], [b[0], b[1], gz], [b[0], b[1], gz + h], [a[0], a[1], gz + h]])
        dirs = corners - np.array([cam['xy'][0], cam['xy'][1], cz])
        u, vv = world_dirs_to_pano_uv(dirs, cam['R'], mirror_u)
        pts = [(float(uu) * 1920, float(w) * small.height) for uu, w in zip(u, vv)]
        # skip polygons wrapping the seam
        if max(p[0] for p in pts) - min(p[0] for p in pts) > 960:
            continue
        colour = (255, 40, 40) if wall['status'] == 'inferred' else (40, 220, 40)
        draw.polygon(pts, outline=colour, width=3)
        label = f"{wall['buildingId'][-6:]}:{wall.get('edge')} d={d:.0f}"
        draw.text((pts[3][0] + 3, pts[3][1] + 3), label, fill=colour)
        labels.append(label)
        if inc >= MIN_INCIDENCE and (nearest is None or d < nearest[0]):
            nearest = (d, wall)
    # horizon line as the pose predicts it
    horizon = []
    for bearing in np.linspace(0, 2 * np.pi, 361):
        u, vv = world_dirs_to_pano_uv(np.array([math.sin(bearing), math.cos(bearing), 0.0]), cam['R'], mirror_u)
        horizon.append((float(u) * 1920, float(vv) * small.height))
    for p in horizon:
        draw.ellipse([p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1], fill=(255, 255, 0))
    tag = f"{'T' if rotation_transpose else 'R'}{'M' if mirror_u else 'U'}"
    overlay = PROBE_DIR / f'{image_id}__overlay__{tag}.jpg'
    small.save(overlay, quality=80)
    result = {'image': image_id, 'overlay': str(overlay), 'wallsDrawn': labels, 'convention': tag}
    if nearest:
        d, wall = nearest
        gz = buildings[wall['buildingId']]['center'][2]
        vp = view_parameters(cam['xy'], wall, gz)
        img, vfov = render_perspective(pano, cam['R'], vp['forward'], vp['hfovDeg'], vp['vfovDeg'], width=1000,
                                       max_height=1100, mirror_u=mirror_u)
        crop = PROBE_DIR / f"{image_id}__{wall['buildingId']}__{wall.get('edge')}__crop__{tag}.jpg"
        Image.fromarray(img).save(crop, quality=80)
        # the opposite direction, for contrast
        img2, _ = render_perspective(pano, cam['R'], vp['forward'] * np.array([-1.0, -1.0, 1.0]), vp['hfovDeg'],
                                     vp['vfovDeg'], width=1000, max_height=1100, mirror_u=mirror_u)
        crop2 = PROBE_DIR / f"{image_id}__{wall['buildingId']}__{wall.get('edge')}__opposite__{tag}.jpg"
        Image.fromarray(img2).save(crop2, quality=80)
        result.update({'nearestWall': {'building': wall['buildingId'], 'edge': wall.get('edge'), 'distanceM': round(d, 2),
                                       **{k: v for k, v in vp.items() if k != 'forward'}},
                       'crop': str(crop), 'opposite': str(crop2)})
    print(json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', nargs='*', default=None, help='building ids (default: prior-only buildings in the ranking\'s 85%% set)')
    ap.add_argument('--budget', type=int, default=56, help='maximum number of crops')
    ap.add_argument('--out', type=pathlib.Path, default=OUT_DIR)
    ap.add_argument('--probe', help='image id: render an orientation probe instead of the crop set')
    ap.add_argument('--rotation-transpose', action='store_true', help='use R^T (convention flip for the probe)')
    ap.add_argument('--mirror-u', action='store_true', help='mirror the panorama column (convention flip for the probe)')
    ap.add_argument('--verdicts', type=pathlib.Path, help='optional JSON {buildingId: verdict} merged into the manifest')
    ap.add_argument('--dry-run', action='store_true', help='select cameras and write the manifest without rendering')
    args = ap.parse_args(argv)
    if args.targets is None:
        args.targets = default_targets()

    t0 = time.time()
    world, frame = load_world()
    buildings = {b['id']: b for b in world['buildings']}
    evidence = json.load(open(EVIDENCE_PATH))
    walls = evidence['facadeWalls']
    walls_by_building: dict[str, list[dict]] = {}
    for w in walls:
        walls_by_building.setdefault(w['buildingId'], []).append(w)
    footprints = load_footprints(frame)
    occ = Occluders(footprints)
    trees = load_trees(frame)
    cams = load_cameras(frame, rotation_transpose=args.rotation_transpose)

    if args.probe:
        run_probe(args.probe, cams, walls, buildings, occ, args.mirror_u, args.rotation_transpose)
        return 0

    missing = [t for t in args.targets if t not in buildings]
    if missing:
        print('unknown building ids:', missing, file=sys.stderr)
        return 2

    per_wall, chosen = select_crops(buildings, args.targets, walls_by_building, cams, occ, trees, args.budget)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    render_seconds = 0.0
    for w, cand, rank in chosen:
        wall = w['wall']
        cam = cams[cand['image']]
        gz = buildings[w['building']]['center'][2]
        vp = view_parameters(cam['xy'], wall, gz)
        name = f"{w['building']}__{w['wallIndex']}__{cand['image']}.jpg"
        path = args.out / name
        size = None
        if not args.dry_run:
            tr = time.time()
            pano = load_pano(cam['path'])
            img, vfov = render_perspective(pano, cam['R'], vp['forward'], vp['hfovDeg'], vp['vfovDeg'],
                                           mirror_u=args.mirror_u)
            Image.fromarray(img).save(path, quality=JPEG_QUALITY)
            size = [int(img.shape[1]), int(img.shape[0])]
            vp['vfovDeg'] = round(vfov, 2)
            render_seconds += time.time() - tr
        entries.append({
            'building': w['building'],
            'wallIndex': w['wallIndex'],
            'wallEdge': wall.get('edge', wall.get('grammarId')),
            'wallStatus': wall['status'],
            'wallLengthM': wall['lengthM'],
            'wallHeightM': wall['heightM'],
            'wallMidpoint': [round(float(x), 2) for x in wall['midpoint']],
            'wallNormal': [round(float(x), 3) for x in wall['normal']],
            'image': cand['image'],
            'captureDate': cam['captureDate'],
            'registrationGrade': cam['registrationGrade'],
            'cameraLocal': cand['cameraLocal'],
            'cameraAltitudeM': cam['altitudeM'],
            'cameraDistanceM': cand['distanceM'],
            'incidence': cand['incidence'],
            'visibleFraction': cand['visibleFraction'],
            'treesNearSightLine': cand['treesNearSightLine'],
            'selectionScore': cand['score'],
            'viewRank': rank,
            'distantFallback': w['fallback'],
            'yawDeg': vp['yawDeg'],
            'pitchDeg': vp['pitchDeg'],
            'hfovDeg': vp['hfovDeg'],
            'vfovDeg': vp['vfovDeg'],
            'wallSpanDeg': vp['wallSpanDeg'],
            'hfovCapped': vp['hfovCapped'],
            'vfovCapped': vp['vfovCapped'],
            'sizePx': size,
            'cropPath': str(path),
        })

    per_building = {}
    for bid in args.targets:
        mine = [w for w in per_wall if w['building'] == bid]
        per_building[bid] = {
            'wallsTotal': len(mine),
            'wallsWithCandidates': sum(bool(w['candidates']) and not w['fallback'] for w in mine),
            'wallsCropped': len({e['wallIndex'] for e in entries if e['building'] == bid}),
            'crops': sum(e['building'] == bid for e in entries),
            'nearestCameraM': min((c['distanceM'] for w in mine for c in w['candidates']), default=None),
            'distantFallback': any(w['fallback'] for w in mine),
            'facadeEvidence': buildings[bid].get('facadeEvidence'),
            'heightM': buildings[bid].get('height'),
        }
    verdicts = json.load(open(args.verdicts)) if args.verdicts else {}
    for bid, rec in per_building.items():
        if bid in verdicts:
            rec['verdict'] = verdicts[bid]
    manifest = {
        'schemaVersion': 1,
        'generatedBy': 'frontage_crops_v15.py',
        'convention': {
            'rotation': 'OpenSfM axis-angle, camera = R @ world (transposed)' if args.rotation_transpose
                        else 'OpenSfM axis-angle, camera = R @ world',
            'panorama': 'u mirrored' if args.mirror_u else 'u = lon/(2pi)+.5 with +z forward at u=.5, +x right; v = .5 - lat/pi',
            'worldFrame': 'x east, y north, z up; UTM 10N metres relative to world.json center',
            'eyeHeightM': EYE_HEIGHT_M,
        },
        'rules': {
            'maxDistanceM': MAX_DISTANCE_M, 'fallbackDistanceM': FALLBACK_DISTANCE_M,
            'preferredDistanceM': list(PREFERRED_DISTANCE_M), 'minIncidence': MIN_INCIDENCE,
            'minVisibleFraction': MIN_VISIBLE_FRACTION, 'fovMargin': FOV_MARGIN, 'maxHfovDeg': MAX_HFOV_DEG,
            'maxVfovDeg': MAX_VFOV_DEG, 'cropWidthPx': CROP_WIDTH_PX, 'budget': args.budget,
        },
        'counts': {
            'targets': len(args.targets), 'wallsConsidered': len(per_wall),
            'wallsWithCandidates': sum(bool(w['candidates']) for w in per_wall),
            'crops': len(entries), 'distinctImages': len({e['image'] for e in entries}),
            'panoramasIndexed': len(cams), 'panoramasOnDisk': sum(c['exists'] for c in cams.values()),
            'footprintOccluders': len(footprints), 'streetTrees': int(len(trees)),
        },
        'perBuilding': per_building,
        'crops': entries,
        'elapsedSeconds': round(time.time() - t0, 2),
        'renderSeconds': round(render_seconds, 2),
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'counts': manifest['counts'], 'perBuilding': per_building,
                      'elapsedSeconds': manifest['elapsedSeconds'], 'renderSeconds': manifest['renderSeconds']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
