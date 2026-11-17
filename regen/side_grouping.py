"""Group connected wall segments that share one measured facade plane."""
import math

import numpy as np


ANGLE_DEG = 1.0
PLANE_M = 0.15
GAP_M = 0.25


def sides_of(walls, base_tolerance=None):
    """Preserve source walls; never move the anchor plane while growing a side.

    Every member's two endpoints must fit that same plane. Revisit remaining
    segments after growth so a connecting segment can join earlier candidates.
    Deterministic longest-first anchors keep input order from changing groups.
    """
    pending = []
    for wall in walls:
        normal = np.asarray(wall['normal'], dtype=float)
        normal = normal / np.linalg.norm(normal)
        tangent = np.array([-normal[1], normal[0]])
        midpoint = np.asarray(wall['midpoint'], dtype=float)
        half = tangent * float(wall['lengthM']) / 2
        endpoints = np.array([midpoint - half, midpoint + half])
        pending.append((wall, normal, tangent, midpoint, endpoints))
    pending.sort(key=lambda item: (-float(item[0]['lengthM']),
                                  *item[3], *item[1], float(item[0]['heightM']), float(item[0].get('zBaseM', 0))))
    sides = []
    cosine = math.cos(math.radians(ANGLE_DEG))
    while pending:
        wall, normal, tangent, midpoint, endpoints = pending.pop(0)
        values = endpoints @ tangent
        side = {'n': normal, 't': tangent, 'v': float(midpoint @ normal),
                'u0': float(values.min()), 'u1': float(values.max()),
                'h': wall['heightM'], 'walls': [wall]}
        if base_tolerance is not None:
            side['zb'] = float(wall['zBaseM'])
        changed = True
        while changed:
            changed = False
            remaining = []
            for item in pending:
                candidate, candidate_normal, _, _, candidate_endpoints = item
                values = candidate_endpoints @ tangent
                lo, hi = float(values.min()), float(values.max())
                gap = max(lo - side['u1'], side['u0'] - hi, 0.)
                coplanar = np.all(np.abs(candidate_endpoints @ normal - side['v']) <= PLANE_M + 1e-9)
                same_base = base_tolerance is None or abs(float(candidate['zBaseM']) - side['zb']) <= base_tolerance
                if (float(candidate_normal @ normal) >= cosine - 1e-12
                        and coplanar and same_base and gap <= GAP_M + 1e-9):
                    side['u0'], side['u1'] = min(side['u0'], lo), max(side['u1'], hi)
                    top = candidate['heightM'] + (float(candidate['zBaseM']) - side['zb'] if base_tolerance is not None else 0)
                    side['h'] = max(side['h'], top)
                    side['walls'].append(candidate)
                    changed = True
                else:
                    remaining.append(item)
            pending = remaining
        sides.append(side)
    return sides
