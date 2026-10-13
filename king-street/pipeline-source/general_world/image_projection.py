"""Small, source independent utilities for projecting validated facade geometry.

The provider pose is deliberately kept separate from registration.  A pose can
be internally consistent while still being in the wrong world frame, so the
public crop operation refuses unregistered cameras unless the caller explicitly
asks for diagnostic output.

Only spherical/equirectangular cameras are supported here.  In particular, an
unknown orientation or a camera model without verified calibration is an error;
this module does not try to infer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, pi
from typing import Any, Mapping

import numpy as np


class Orientation(IntEnum):
    """Provider orientation values understood by this first implementation."""

    EQUIRECTANGULAR = 1


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def angle_axis_matrix(rotation_angle_axis: Any) -> np.ndarray:
    """Return a world-to-camera matrix from a finite angle-axis vector.

    The zero vector is the identity rotation.  Any finite angle is accepted:
    angle-axis vectors are periodic and callers may use the representation
    emitted by their reconstruction system.
    """

    r = _vector(rotation_angle_axis, "rotation_angle_axis")
    theta = float(np.linalg.norm(r))
    if theta == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = r / theta
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _orientation(value: Any) -> Orientation:
    # bool is an int subclass, but is not an orientation enum value.
    if isinstance(value, bool):
        raise ValueError("unsupported spherical orientation; expected orientation 1")
    try:
        return Orientation(int(value))
    except (TypeError, ValueError):
        raise ValueError("unsupported spherical orientation; expected orientation 1") from None


@dataclass(frozen=True)
class CameraTransform:
    """Validated world-to-camera transform for a spherical camera.

    ``rotation_angle_axis`` and ``translation`` use the reconstruction
    convention ``camera_point = R @ world_point + translation``.  The camera
    centre in world coordinates is therefore ``-R.T @ translation``.
    """

    rotation_angle_axis: np.ndarray
    translation: np.ndarray
    orientation: Orientation = Orientation.EQUIRECTANGULAR
    world_registration_validated: bool = False
    diagnostic: bool = False

    def __post_init__(self) -> None:
        r = _vector(self.rotation_angle_axis, "rotation_angle_axis")
        t = _vector(self.translation, "translation")
        ori = _orientation(self.orientation)
        if not isinstance(self.world_registration_validated, (bool, np.bool_)):
            raise ValueError("world_registration_validated must be boolean")
        if not isinstance(self.diagnostic, (bool, np.bool_)):
            raise ValueError("diagnostic must be boolean")
        object.__setattr__(self, "rotation_angle_axis", r)
        object.__setattr__(self, "translation", t)
        object.__setattr__(self, "orientation", ori)
        object.__setattr__(self, "world_registration_validated", bool(self.world_registration_validated))
        object.__setattr__(self, "diagnostic", bool(self.diagnostic))
        # Compute once so malformed rotations fail at construction time.
        angle_axis_matrix(r)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return angle_axis_matrix(self.rotation_angle_axis)

    @property
    def camera_center(self) -> np.ndarray:
        return -self.rotation_matrix.T @ self.translation

    @classmethod
    def from_metadata(cls, camera: Mapping[str, Any], *, diagnostic: bool = False) -> "CameraTransform":
        """Build a transform from a normalized ``camera`` metadata mapping.

        ``diagnostic=True`` permits an unregistered provider pose, but marks the
        resulting transform as diagnostic so it cannot be mistaken for admitted
        evidence by callers.
        """

        if not isinstance(camera, Mapping):
            raise ValueError("camera metadata must be a mapping")
        model = camera.get("model")
        if str(model).lower() not in {"spherical", "equirectangular"}:
            raise ValueError("only spherical cameras are supported")
        if "rotationAngleAxis" in camera:
            rotation = camera["rotationAngleAxis"]
        else:
            rotation = camera.get("rotation_angle_axis")
        translation = camera.get("translation")
        if rotation is None or translation is None:
            raise ValueError("camera metadata requires rotationAngleAxis and translation")
        registered = camera.get("worldRegistrationValidated", camera.get("world_registration_validated"))
        if not isinstance(registered, (bool, np.bool_)):
            raise ValueError("camera metadata requires boolean worldRegistrationValidated")
        if not bool(registered) and not diagnostic:
            raise ValueError("world registration is not validated; use diagnostic=True explicitly")
        orientation = camera.get("orientation", Orientation.EQUIRECTANGULAR)
        return cls(rotation, translation, orientation, bool(registered), diagnostic=diagnostic)


def camera_vectors_to_uv(vectors_camera: Any) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame vectors to normalized spherical coordinates."""

    vectors = np.asarray(vectors_camera, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or not np.all(np.isfinite(vectors)):
        raise ValueError("camera vectors must have a finite final dimension of length 3")
    x, y, z = np.moveaxis(vectors, -1, 0)
    horizontal = np.sqrt(x * x + z * z)
    length = np.sqrt(horizontal * horizontal + y * y)
    if np.any(length == 0.0):
        raise ValueError("camera vectors must be nonzero")
    # Keep the exact normalized value, including the valid endpoint u=1 for
    # atan2(...)=pi.  Image sampling wraps this coordinate at its seam.
    u = np.arctan2(x, z) / (2.0 * pi) + 0.5
    v = np.arctan2(y, horizontal) / pi + 0.5
    return u, v


def project_world_points(points_world: Any, camera: CameraTransform, *, diagnostic: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
    """Project world points with a caller-supplied validated world frame."""

    if not camera.world_registration_validated and not (diagnostic or camera.diagnostic):
        raise ValueError("world registration is not validated")
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("world points must have a finite final dimension of length 3")
    vectors = np.einsum("ij,...j->...i", camera.rotation_matrix, points) + camera.translation
    u, v = camera_vectors_to_uv(vectors)
    return u, v, bool(camera.world_registration_validated)


def bilinear_sample_equirectangular(image: Any, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
    """Sample an equirectangular image, wrapping only at its horizontal seam.

    Samples outside the vertical image range are invalid.  A pixel is valid
    only when all four bilinear neighbours are finite, so missing source pixels
    remain holes instead of being filled with invented colour.
    """

    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.shape[0] < 1 or source.shape[1] < 1:
        raise ValueError("image must be a non-empty HxW or HxWxC array")
    if source.ndim == 3 and source.shape[2] < 1:
        raise ValueError("image must have at least one channel")
    uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
    finite = np.isfinite(uu) & np.isfinite(vv) & (vv >= 0.0) & (vv <= 1.0)
    h, w = source.shape[:2]
    # Pixel centres are x + .5.  This makes the seam continuous at u=0/1.
    # Keep invalid coordinates finite while computing array indices.  The
    # original finite mask is retained below, so this cannot turn NaN/Inf into
    # a fabricated valid sample or trigger an unsafe integer cast.
    safe_u = np.where(np.isfinite(uu), uu, 0.0)
    safe_v = np.where(np.isfinite(vv), vv, 0.0)
    xf = np.mod(safe_u * w - 0.5, w)
    yf = np.clip(safe_v * (h - 1), 0.0, h - 1.0)
    x0 = np.floor(xf).astype(np.int64)
    x1 = (x0 + 1) % w
    y0 = np.floor(yf).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = xf - x0
    wy = yf - y0
    # Source NaNs are holes; infinities are holes too.
    a, b = source[y0, x0], source[y0, x1]
    c, d = source[y1, x0], source[y1, x1]
    source_finite = np.isfinite(a).all(axis=-1) if source.ndim == 3 else np.isfinite(a)
    source_finite &= np.isfinite(b).all(axis=-1) if source.ndim == 3 else np.isfinite(b)
    source_finite &= np.isfinite(c).all(axis=-1) if source.ndim == 3 else np.isfinite(c)
    source_finite &= np.isfinite(d).all(axis=-1) if source.ndim == 3 else np.isfinite(d)
    valid = finite & source_finite
    weight_x = wx[..., None] if source.ndim == 3 else wx
    weight_y = wy[..., None] if source.ndim == 3 else wy
    result = ((1.0 - weight_y) * ((1.0 - weight_x) * a + weight_x * b) + weight_y * ((1.0 - weight_x) * c + weight_x * d)).astype(np.float64)
    result = np.where(valid[..., None], result, 0.0) if source.ndim == 3 else np.where(valid, result, 0.0)
    if np.issubdtype(source.dtype, np.integer):
        result = np.rint(result).clip(np.iinfo(source.dtype).min, np.iinfo(source.dtype).max).astype(source.dtype)
    else:
        result = result.astype(source.dtype, copy=False)
    return result, valid


@dataclass(frozen=True)
class FacadeCrop:
    rgb: np.ndarray
    valid_mask: np.ndarray
    front_facing_mask: np.ndarray
    grazing_mask: np.ndarray
    registered: bool
    diagnostic: bool
    u: np.ndarray
    v: np.ndarray
    occlusion_mask: np.ndarray
    occlusion_checked: bool
    occlusion_unchecked: bool
    admitted: bool

    @property
    def provenance_flags(self) -> dict[str, bool]:
        """Machine-readable evidence flags for downstream admission checks."""

        return {
            "worldRegistrationValidated": self.registered,
            "diagnostic": self.diagnostic,
            "occlusionChecked": self.occlusion_checked,
            "occlusionUnchecked": self.occlusion_unchecked,
            "admitted": self.admitted,
            # Projection admission is an evidence-routing flag only.  It does
            # not claim measured facade geometry or inferred openings.
            "facadeMeasurementCertified": False,
            "openingInferenceCertified": False,
        }

    @property
    def provenance(self) -> dict[str, bool]:
        # Short alias used by callers serializing crop provenance.
        return self.provenance_flags


def project_facade_crop(
    image: Any,
    camera: CameraTransform,
    origin: Any,
    horizontal: Any,
    vertical: Any,
    width_m: float,
    height_m: float,
    pixels_per_m: float,
    *,
    max_dimension: int = 8192,
    max_pixels: int = 16_777_216,
    grazing_cosine: float = 0.05,
    source_occlusion_mask: Any | None = None,
    diagnostic: bool = False,
) -> FacadeCrop:
    """Project a metric rectangle into an RGB crop with explicit masks.

    ``source_occlusion_mask`` is an image-sized boolean array where ``True``
    means the source pixel is known to be occluded (for example by a car or a
    tree).  With no mask, projection validity is still returned for geometry
    diagnostics, but provenance is ``occlusionUnchecked`` and ``admitted`` is
    false.  A geometric projection never implies an unobstructed facade view.
    """

    if not camera.world_registration_validated and not (diagnostic or camera.diagnostic):
        raise ValueError("world registration is not validated; admitted output is refused")
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError("image must be an HxWxRGB/RGBA array")
    o = _vector(origin, "origin")
    h = _vector(horizontal, "horizontal")
    wv = _vector(vertical, "vertical")
    hn, vn = np.linalg.norm(h), np.linalg.norm(wv)
    if hn == 0.0 or vn == 0.0:
        raise ValueError("horizontal and vertical vectors must be nonzero")
    h, wv = h / hn, wv / vn
    if abs(float(np.dot(h, wv))) > 1e-6:
        raise ValueError("horizontal and vertical vectors must be orthogonal")
    normal = np.cross(h, wv)
    nn = np.linalg.norm(normal)
    if nn == 0.0:
        raise ValueError("horizontal and vertical vectors must not be parallel")
    normal /= nn
    width_m, height_m, pixels_per_m = map(float, (width_m, height_m, pixels_per_m))
    if not all(np.isfinite(x) and x > 0.0 for x in (width_m, height_m, pixels_per_m)):
        raise ValueError("width_m, height_m, and pixels_per_m must be positive finite values")
    if not isinstance(max_dimension, (int, np.integer)) or max_dimension < 1:
        raise ValueError("max_dimension must be a positive integer")
    if not isinstance(max_pixels, (int, np.integer)) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if not np.isfinite(grazing_cosine) or not 0.0 <= grazing_cosine <= 1.0:
        raise ValueError("grazing_cosine must be between zero and one")
    cols, rows = ceil(width_m * pixels_per_m), ceil(height_m * pixels_per_m)
    if cols > max_dimension or rows > max_dimension or cols * rows > max_pixels:
        raise ValueError("facade crop exceeds image dimension limits")

    # Pixel centres avoid putting an entire crop on a rectangle boundary.
    xs = (np.arange(cols, dtype=np.float64) + 0.5) / cols * width_m
    ys = (np.arange(rows, dtype=np.float64) + 0.5) / rows * height_m
    points = o + xs[None, :, None] * h + ys[:, None, None] * wv
    vectors = np.einsum("ij,hwj->hwi", camera.rotation_matrix, points) + camera.translation
    u, v = camera_vectors_to_uv(vectors)
    rgb, source_valid = bilinear_sample_equirectangular(image_array[..., :3], u, v)
    h_image, w_image = image_array.shape[:2]
    occlusion_checked = source_occlusion_mask is not None
    if source_occlusion_mask is None:
        occlusion = np.zeros((rows, cols), dtype=bool)
        unoccluded = np.ones((rows, cols), dtype=bool)
    else:
        mask = np.asarray(source_occlusion_mask)
        if mask.shape != (h_image, w_image) or mask.dtype.kind not in "bui":
            raise ValueError("source_occlusion_mask must be an image-sized boolean or integer array")
        # Treat every bilinear neighbour marked true as occluded.  This keeps
        # an occlusion boundary conservative instead of inventing a clear
        # sample from a mix of blocked and unblocked source pixels.
        safe_u = np.where(np.isfinite(u), u, 0.0)
        safe_v = np.where(np.isfinite(v), v, 0.0)
        xf = np.mod(safe_u * w_image - 0.5, w_image)
        yf = np.clip(safe_v * (h_image - 1), 0.0, h_image - 1.0)
        x0 = np.floor(xf).astype(np.int64)
        x1 = (x0 + 1) % w_image
        y0 = np.floor(yf).astype(np.int64)
        y1 = np.minimum(y0 + 1, h_image - 1)
        occlusion = mask[y0, x0].astype(bool) | mask[y0, x1].astype(bool) | mask[y1, x0].astype(bool) | mask[y1, x1].astype(bool)
        unoccluded = ~occlusion
    camera_to_points = points - camera.camera_center
    lengths = np.linalg.norm(camera_to_points, axis=-1)
    cos_to_camera = np.einsum("j,hwj->hw", normal, -camera_to_points) / np.maximum(lengths, np.finfo(float).tiny)
    front = cos_to_camera > 0.0
    grazing = np.abs(cos_to_camera) <= float(grazing_cosine)
    valid = source_valid & front & ~grazing & unoccluded
    registered = bool(camera.world_registration_validated)
    diagnostic_output = bool(diagnostic or camera.diagnostic)
    # A crop can only be admitted once both world registration and occlusion
    # evidence are present, and diagnostic output is always non-admitted.
    admitted = registered and occlusion_checked and not diagnostic_output
    return FacadeCrop(
        rgb,
        valid,
        front,
        grazing,
        registered,
        diagnostic_output,
        u,
        v,
        occlusion,
        occlusion_checked,
        not occlusion_checked,
        admitted,
    )


__all__ = [
    "CameraTransform",
    "FacadeCrop",
    "Orientation",
    "angle_axis_matrix",
    "bilinear_sample_equirectangular",
    "camera_vectors_to_uv",
    "project_facade_crop",
    "project_world_points",
]
