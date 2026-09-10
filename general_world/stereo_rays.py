"""Small, bounded stereo-ray primitives for diagnostic facade hypotheses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _vec(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (3,) and np.isfinite(result).all() else None


def _unit(value: Any) -> np.ndarray | None:
    result = _vec(value)
    if result is None:
        return None
    length = float(np.linalg.norm(result))
    return result / length if length > 1e-9 else None


def closest_positive_rays(
    camera_a: Any,
    direction_a: Any,
    camera_b: Any,
    direction_b: Any,
    *,
    min_angle_deg: float = 5.0,
    max_condition: float = 100.0,
    max_residual_m: float = 1.0,
) -> dict[str, Any]:
    """Find the closest positive points on two rays, with bounded geometry."""
    a, b = _vec(camera_a), _vec(camera_b)
    da, db = _unit(direction_a), _unit(direction_b)
    if any(value is None for value in (a, b, da, db)):
        return {"valid": False, "reason": "nonfinite_or_invalid_ray"}
    cosine = float(np.clip(np.dot(da, db), -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if angle < min_angle_deg:
        return {"valid": False, "reason": "parallel_or_low_parallax", "angleDeg": angle}
    matrix = np.column_stack((da, -db))
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return {"valid": False, "reason": "ill_conditioned", "angleDeg": angle}
    if not np.isfinite(condition) or condition > max_condition:
        return {"valid": False, "reason": "ill_conditioned", "angleDeg": angle, "condition": condition}
    try:
        depths, _, _, _ = np.linalg.lstsq(matrix, b - a, rcond=None)
    except np.linalg.LinAlgError:
        return {"valid": False, "reason": "ill_conditioned", "angleDeg": angle, "condition": condition}
    ta, tb = map(float, depths)
    if not np.isfinite([ta, tb]).all() or ta <= 0.0 or tb <= 0.0:
        return {"valid": False, "reason": "negative_or_zero_depth", "angleDeg": angle, "condition": condition, "depthA": ta, "depthB": tb}
    point_a, point_b = a + ta * da, b + tb * db
    residual = float(np.linalg.norm(point_a - point_b))
    if not np.isfinite(residual) or residual > max_residual_m:
        return {"valid": False, "reason": "ray_residual_too_large", "angleDeg": angle, "condition": condition, "depthA": ta, "depthB": tb, "rayResidualM": residual}
    return {"valid": True, "angleDeg": angle, "condition": condition, "depthA": ta, "depthB": tb, "pointA": point_a, "pointB": point_b, "center": (point_a + point_b) * 0.5, "rayResidualM": residual}


def metric_plane_point(record: dict[str, Any], pixel_xy: Any, image_size: tuple[int, int]) -> np.ndarray:
    """Map a pixel centre onto the record's assumed facade plane."""
    origin = _vec(record.get("metricFrame", {}).get("origin"))
    horizontal = _unit(record.get("metricFrame", {}).get("horizontal"))
    vertical = _unit(record.get("metricFrame", {}).get("vertical"))
    if origin is None or horizontal is None or vertical is None:
        raise ValueError("invalid metric frame")
    try:
        x, y = map(float, pixel_xy)
        width, height = map(float, image_size)
        width_m, height_m = float(record["widthM"]), float(record["diagnosticHeightM"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid metric dimensions") from None
    if not np.isfinite([x, y, width, height, width_m, height_m]).all() or width <= 0 or height <= 0 or width_m <= 0 or height_m <= 0:
        raise ValueError("invalid metric dimensions")
    return origin + horizontal * (x / width * width_m) + vertical * ((height - y) / height * height_m)


def ray_for_pixel(record: dict[str, Any], pixel_xy: Any, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    camera = _vec(record.get("cameraCenterWorldM"))
    if camera is None:
        raise ValueError("invalid camera center")
    target = metric_plane_point(record, pixel_xy, image_size)
    direction = target - camera
    unit_direction = _unit(direction)
    if unit_direction is None:
        raise ValueError("camera lies on target")
    return camera, unit_direction


def triangulate_pixel_pair(
    left: dict[str, Any],
    left_pixel: Any,
    left_size: tuple[int, int],
    right: dict[str, Any],
    right_pixel: Any,
    right_size: tuple[int, int],
    *,
    normal_limit_m: float = 5.0,
    panel_margin_m: float = 2.0,
    **ray_options: Any,
) -> dict[str, Any]:
    """Triangulate two proposal centres and apply only bounded prior checks."""
    try:
        camera_a, direction_a = ray_for_pixel(left, left_pixel, left_size)
        camera_b, direction_b = ray_for_pixel(right, right_pixel, right_size)
        plane_a = metric_plane_point(left, left_pixel, left_size)
        plane_b = metric_plane_point(right, right_pixel, right_size)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc)}
    result = closest_positive_rays(camera_a, direction_a, camera_b, direction_b, **ray_options)
    if not result.get("valid"):
        return result
    center = np.asarray(result["center"], dtype=float)
    origin = _vec(left["metricFrame"]["origin"])
    horizontal, vertical = _unit(left["metricFrame"]["horizontal"]), _unit(left["metricFrame"]["vertical"])
    normal = _unit(np.cross(horizontal, vertical)) if horizontal is not None and vertical is not None else None
    if origin is None or horizontal is None or vertical is None or normal is None:
        return {"valid": False, "reason": "invalid_metric_frame"}
    local_x, local_y = float(np.dot(center - origin, horizontal)), float(np.dot(center - origin, vertical))
    try:
        width_m, height_m = float(left["widthM"]), float(left["diagnosticHeightM"])
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "reason": "invalid_metric_dimensions"}
    offset = float(np.dot(center - origin, normal))
    if abs(offset) > normal_limit_m:
        return {"valid": False, "reason": "normal_departure_too_large", "signedOffsetFromAssumedPlaneM": offset, "rayResidualM": result["rayResidualM"]}
    if local_x < -panel_margin_m or local_x > width_m + panel_margin_m or local_y < -panel_margin_m or local_y > height_m + panel_margin_m:
        return {"valid": False, "reason": "outside_panel_margin", "signedOffsetFromAssumedPlaneM": offset, "rayResidualM": result["rayResidualM"]}
    return {**result, "valid": True, "center": center, "signedOffsetFromAssumedPlaneM": offset, "planarReprojectionErrorM": float(np.linalg.norm(plane_a - plane_b)), "panelLocalXYM": [local_x, local_y]}

