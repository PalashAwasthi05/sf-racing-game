"""Review helpers for inspecting production tree-sizing decisions."""
import math

import numpy as np

from pipeline.tree_sizing import (
    assess_lidar_height, deduplicate_inventory_records, fallback_height_m,
    inventory_identity, trunk_radius,
)


def project_point(point_xyz, camera):
    """Project source-world XYZ into a manifest camera's pixel frame."""
    origin = np.asarray(camera["from"], dtype=float)
    target = np.asarray(camera["at"], dtype=float)
    forward = target - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    v = np.asarray(point_xyz, dtype=float) - origin
    depth = float(v @ forward)
    if depth <= 1e-8: return None
    focal = float(camera["height"]) / (2.0 * math.tan(math.radians(float(camera["vfovDeg"])) / 2.0))
    return [float(camera["width"]) / 2.0 + focal * float(v @ right) / depth,
            float(camera["height"]) / 2.0 - focal * float(v @ up) / depth, depth]


def project_trunk_quad(center_xy, base_z, height_m, radius_m, camera):
    """Four projected trunk corners for a same-camera geometry A/B overlay."""
    origin = np.asarray(camera["from"], dtype=float)
    forward = np.asarray(camera["at"], dtype=float) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    x, y = map(float, center_xy)
    corners = [[x - radius_m * right[0], y - radius_m * right[1], base_z],
               [x + radius_m * right[0], y + radius_m * right[1], base_z],
               [x + radius_m * right[0], y + radius_m * right[1], base_z + height_m],
               [x - radius_m * right[0], y - radius_m * right[1], base_z + height_m]]
    return [project_point(corner, camera) for corner in corners]
