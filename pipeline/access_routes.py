"""Evidence-backed exterior access routes in a facade frame.

Coordinates are measured in metres: x along the wall, z above the wall-frame
ground datum, and depth outward from the wall. An ordered route contains only
observed flights and landings. Its x coordinates may lie outside the host wall
interval; they are never clipped or inferred from terrain.
"""
import math
import numpy as np
from shapely.geometry import Polygon


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite(v) for v in value):
        return None
    return [float(v) for v in value]


def _serves(value):
    """Normalize explicit source-array owners so unresolved routes suppress only their own legacy flight."""
    items = [value] if isinstance(value, dict) else value
    if not isinstance(items, list) or not items:
        return None
    out = []
    for item in items:
        if (not isinstance(item, dict) or item.get('collection') not in ('porches', 'openings')
                or not isinstance(item.get('index'), int) or isinstance(item.get('index'), bool)
                or item['index'] < 0):
            return None
        target = {'collection': item['collection'], 'index': item['index']}
        if target not in out: out.append(target)
    return out


def compile_access_routes(raw_routes):
    """Validate measured route segments without completing or snapping them.

    Invalid routes are returned as unresolved records with a reason and no
    geometry. A route marked ``continuation: 'occluded'`` may still contain a
    fully measured visible prefix; only that prefix is compiled.
    """
    compiled, errors = [], []
    for index, raw in enumerate(raw_routes or []):
        reason = None
        if not isinstance(raw, dict):
            compiled.append({'status': 'unresolved', 'segments': [], 'reason': 'route must be an object'})
            errors.append((index, 'route must be an object'))
            continue
        serves = _serves(raw.get('serves'))
        status = raw.get('status')
        segments = raw.get('segments')
        if raw.get('needsFreshSourceCoordinates'):
            route = {'status': 'unresolved', 'segments': [], 'needsFreshSourceCoordinates': True,
                     'reason': raw.get('note') or 'route coordinates need re-reading after camera remap'}
            if serves is not None: route['serves'] = serves
            else: route['reason'] += '; an explicit porch/opening owner is required'
            compiled.append(route)
            continue
        if status in ('unresolved', 'occluded'):
            route = {'status': 'unresolved', 'segments': [],
                     'reason': raw.get('note') or ('route is occluded' if status == 'occluded' else 'route measurements unresolved')}
            if serves is not None: route['serves'] = serves
            else: route['reason'] += '; an explicit porch/opening owner is required'
            if status == 'occluded' or raw.get('continuation') == 'occluded': route['continuation'] = 'occluded'
            compiled.append(route)
            continue
        if serves is None:
            reason = 'route requires explicit serves collection/index'
        elif status != 'observed' or not isinstance(segments, list) or not segments:
            reason = 'route requires observed status and at least one measured segment'
        normalized = []
        if reason is None:
            for si, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    reason = f'segment {si} must be an object'; break
                kind = segment.get('kind')
                if kind == 'flight':
                    start, end = _point(segment.get('from')), _point(segment.get('to'))
                    steps, width = segment.get('steps'), segment.get('widthM')
                    if start is None or end is None:
                        reason = f'flight {si} requires finite from/to [xM,zM,depthM] endpoints'; break
                    if (not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 80
                            or not _finite(width) or not .4 <= float(width) <= 10.):
                        reason = f'flight {si} requires measured steps and widthM'; break
                    construction = segment.get('construction', 'closed')
                    if construction not in ('closed', 'waist', 'open_risers'):
                        reason = f'flight {si} construction must be closed, waist, or open_risers'; break
                    tread_thickness = segment.get('treadThicknessM')
                    if construction == 'open_risers' and (not _finite(tread_thickness) or not .04 <= float(tread_thickness) <= .3):
                        reason = f'open-riser flight {si} requires measured treadThicknessM'; break
                    plan_length = math.hypot(end[0] - start[0], end[2] - start[2])
                    if plan_length < .1 or abs(end[1] - start[1]) < .05:
                        reason = f'flight {si} needs distinct plan endpoints and a measured elevation change'; break
                    normalized.append({'kind': 'flight', 'from': start, 'to': end,
                                       'steps': steps, 'widthM': float(width),
                                       'construction': construction,
                                       **({'treadThicknessM': float(tread_thickness)} if construction == 'open_risers' else {}),
                                       'material': segment.get('material', raw.get('material', '#817d76'))})
                elif kind == 'landing':
                    outline = segment.get('outline'); z = segment.get('zM'); thick = segment.get('thicknessM')
                    try:
                        poly = Polygon(outline)
                    except (TypeError, ValueError):
                        poly = Polygon()
                    if (not isinstance(outline, (list, tuple)) or len(outline) < 3
                            or any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(_finite(v) for v in p) for p in outline)
                            or poly.is_empty
                            or not poly.is_valid or poly.area < .04
                            or not _finite(z) or not _finite(thick) or not .04 <= float(thick) <= .6):
                        reason = f'landing {si} requires a measured simple outline, zM, and thicknessM'; break
                    normalized.append({'kind': 'landing', 'outline': [[float(x), float(d)] for x, d in outline],
                                       'zM': float(z), 'thicknessM': float(thick),
                                       'material': segment.get('material', raw.get('material', '#817d76'))})
                else:
                    reason = f'segment {si} kind must be flight or landing'; break
        if reason:
            route = {'status': 'unresolved', 'segments': [], 'reason': reason}
            if serves is not None: route['serves'] = serves
            compiled.append(route)
            errors.append((index, reason))
        else:
            route = {'status': 'observed', 'segments': normalized, 'serves': serves}
            if raw.get('continuation') in ('occluded', 'unresolved'):
                route['continuation'] = raw['continuation']
            compiled.append(route)
    return compiled, errors


def route_layout(route):
    """Return solid tread boxes and landing slabs for already validated route data."""
    if not isinstance(route, dict) or route.get('status') != 'observed':
        return []
    layout = []
    for segment in route.get('segments', []):
        if segment['kind'] == 'landing':
            layout.append({'kind': 'landing', 'outline': segment['outline'],
                           'bottomZ': segment['zM'] - segment['thicknessM'], 'topZ': segment['zM'],
                           'material': segment['material']})
            continue
        x0, z0, d0 = segment['from']; x1, z1, d1 = segment['to']
        delta = np.array([x1 - x0, d1 - d0], dtype=float)
        length = float(np.linalg.norm(delta)); tangent = delta / length
        lateral = np.array([-tangent[1], tangent[0]])
        count = int(segment['steps']); half = float(segment['widthM']) / 2
        construction = segment.get('construction', 'closed')
        if construction in ('closed', 'waist'):
            section = [[0., z0]]
            dz = z1 - z0
            for i in range(1, count + 1):
                distance = length * i / count
                level = z0 + dz * (i - 1) / count
                next_level = z0 + dz * i / count
                section.extend([[distance, level], [distance, next_level]])
            bottom = min(z0, z1) - .1
            section.extend([[length, bottom], [0., bottom]])
            layout.append({'kind': 'flight-core', 'crossSection': section,
                           'origin': [x0, d0], 'tangent': tangent.tolist(), 'lateral': lateral.tolist(),
                           'widthM': 2 * half, 'material': segment['material']})
            continue
        for i in range(count):
            a, b = i / count, (i + 1) / count
            pa = np.array([x0, d0]) + delta * a
            pb = np.array([x0, d0]) + delta * b
            corners = [pa - lateral * half, pb - lateral * half,
                       pb + lateral * half, pa + lateral * half]
            dz = z1 - z0
            if dz < 0:
                top = z0 + dz * a
            else:
                top = z0 + dz * b
            bottom = top - float(segment['treadThicknessM'])
            layout.append({'kind': 'tread', 'outline': [[float(x), float(d)] for x, d in corners],
                           'bottomZ': bottom, 'topZ': top,
                           'material': segment['material'], 'stepIndex': i})
    return layout


def draw_route(route, point, put, key='detail'):
    """Extrude a validated route's tread and landing solids into the facade frame."""
    if not isinstance(route, dict) or route.get('status') != 'observed':
        return 0
    import trimesh
    drawn = 0
    for item in route_layout(route):
        if item['kind'] == 'flight-core':
            mesh = trimesh.creation.extrude_polygon(Polygon(item['crossSection']), height=item['widthM'], engine='earcut')
            x0, d0 = item['origin']; tangent = np.asarray(item['tangent']); lateral = np.asarray(item['lateral'])
            half = item['widthM'] / 2
            mesh.vertices = np.asarray([
                point(x0 + tangent[0] * float(v[0]) + lateral[0] * (float(v[2]) - half),
                      float(v[1]),
                      d0 + tangent[1] * float(v[0]) + lateral[1] * (float(v[2]) - half))
                for v in mesh.vertices])
            mesh.fix_normals()
            put(mesh, key, item['material'])
            drawn += 1
            continue
        bottom, top = item['bottomZ'], item['topZ']
        if top - bottom <= 1e-6:
            continue
        mesh = trimesh.creation.extrude_polygon(Polygon(item['outline']), height=top - bottom, engine='earcut')
        mesh.vertices = np.asarray([point(float(v[0]), bottom + float(v[2]), float(v[1])) for v in mesh.vertices])
        mesh.fix_normals()
        put(mesh, key, item['material'])
        drawn += 1
    return drawn
