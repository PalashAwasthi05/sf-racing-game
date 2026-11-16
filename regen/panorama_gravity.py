"""Estimate camera-space world-up from vertical architectural lines in an equirectangular panorama.

No camera rotation is used during detection. A world-vertical line and the camera center define a plane containing
world-up; endpoint-ray great-circle normals from those lines therefore share a perpendicular direction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


_NOMINAL_UP = np.array([0., -1., 0.])


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("expected a finite nonzero vector")
    return vector / length


def _line_observation(ray_a, ray_b, sector, minimum_deg=3.):
    """Return (great-circle normal, capped angular-length weight, azimuth sector), or None."""
    a, b = _unit(ray_a), _unit(ray_b)
    arc = math.degrees(math.acos(float(np.clip(a @ b, -1., 1.))))
    if not np.isfinite(arc) or arc < minimum_deg:
        return None
    normal = np.cross(a, b)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    # Long roof/cornice runs carry more information, but cannot dominate the fit.
    weight = float(np.clip(arc / 15., .2, 1.))
    return normal / norm, weight, int(sector)


def _fit_gravity(normals, weights=None, sectors=None, *, prior=_NOMINAL_UP,
                 max_prior_angle_deg=35., inlier_tolerance_deg=2.5, min_support=20,
                 min_sectors=3, min_inlier_fraction=.14, min_score_margin=.08,
                 min_eigen_gap=.04, ransac_iterations=768, groups=None,
                 _validate_holdout=True):
    """Fit vertical-line plane normals with capped sector-weighted RANSAC and evidence checks."""
    normals = np.asarray(normals, dtype=float).reshape((-1, 3))
    prior = _unit(prior)
    weights = np.ones(len(normals), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    sectors = np.zeros(len(normals), dtype=int) if sectors is None else np.asarray(sectors, dtype=int)
    groups = None if groups is None else np.asarray(groups, dtype=int)
    valid = (np.isfinite(normals).all(axis=1) & np.isfinite(weights) & (weights > 0))
    if groups is not None:
        valid &= np.isfinite(groups)
    normals, weights, sectors = normals[valid], weights[valid], sectors[valid]
    if groups is not None:
        groups = groups[valid]
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10
    normals, weights, sectors = normals[valid] / lengths[valid, None], weights[valid], sectors[valid]
    if groups is not None:
        groups = groups[valid]
    empty = {'gravity': None, 'accepted': False, 'segment_count': int(len(normals)), 'support': 0,
             'support_sectors': 0, 'support_fraction': 0., 'angular_residuals_deg': [],
             'reason': 'insufficient line evidence'}
    if len(normals) < 2:
        return empty

    prior_limit = math.radians(float(max_prior_angle_deg))
    tolerance = math.sin(math.radians(float(inlier_tolerance_deg)))
    candidates = [prior]
    rng = np.random.default_rng(0)
    pairs = min(int(ransac_iterations), max(0, len(normals) * (len(normals) - 1) // 2))
    if pairs:
        for i, j in rng.integers(0, len(normals), size=(pairs, 2)):
            if i == j:
                continue
            candidate = np.cross(normals[i], normals[j])
            magnitude = float(np.linalg.norm(candidate))
            if magnitude < 1e-7:
                continue
            candidate /= magnitude
            if candidate @ prior < 0:
                candidate = -candidate
            if float(np.arccos(np.clip(candidate @ prior, -1., 1.))) <= prior_limit:
                candidates.append(candidate)

    sector_cap = max(6., min(30., math.sqrt(len(normals)) * .65))

    def inliers(candidate):
        residual = np.abs(normals @ candidate)
        return residual <= tolerance

    def support_score(candidate):
        mask = inliers(candidate)
        active, counts = np.unique(sectors[mask], return_counts=True)
        if len(active) < int(min_sectors):
            return None
        sector_weight = np.array([weights[mask & (sectors == sector)].sum() for sector in active])
        # Cap each azimuth bin while retaining its actual support; each bin is not assigned an equal fixed vote.
        return (float(np.minimum(sector_weight, sector_cap).sum()), int(mask.sum()), len(active))

    def refine(seed):
        estimate = seed
        for _ in range(5):
            mask = inliers(estimate)
            active, counts = np.unique(sectors[mask], return_counts=True)
            if int(mask.sum()) < 2 or len(active) < int(min_sectors):
                break
            scales = {}
            for sector, count in zip(active, counts):
                sector_sum = weights[mask & (sectors == sector)].sum()
                scales[sector] = min(1., sector_cap / max(sector_sum, 1e-12))
            w = np.array([weights[k] * scales.get(sectors[k], 0.) if mask[k] else 0.
                          for k in range(len(normals))])
            covariance = (normals * w[:, None]).T @ normals
            _, vectors = np.linalg.eigh(covariance)
            updated = vectors[:, 0]
            if updated @ prior < 0:
                updated = -updated
            if float(np.arccos(np.clip(updated @ prior, -1., 1.))) > prior_limit:
                break
            estimate = updated
        return estimate

    ranked = []
    for seed in candidates:
        estimate = refine(seed)
        candidate_score = support_score(estimate)
        if candidate_score is not None:
            ranked.append((candidate_score, estimate))
    if not ranked:
        empty['reason'] = 'no gravity consensus spans enough azimuth sectors'
        return empty
    ranked.sort(key=lambda row: (row[0][0], row[0][1]), reverse=True)
    # Collapse nearby RANSAC hypotheses and compare the best peak with distinct alternatives.
    peaks = []
    for candidate_score, estimate in ranked:
        if all(float(np.degrees(np.arccos(np.clip(estimate @ other, -1., 1.)))) > 3. for _, other in peaks):
            peaks.append((candidate_score, estimate))
    best_score, best = peaks[0]
    runner_score = peaks[1][0][0] if len(peaks) > 1 else 0.
    score_margin = (best_score[0] - runner_score) / max(best_score[0], 1e-9)

    residual = np.arcsin(np.clip(np.abs(normals @ best), 0., 1.))
    mask = residual <= math.radians(float(inlier_tolerance_deg))
    sectors_hit = np.unique(sectors[mask])
    count, sector_count = int(mask.sum()), int(len(sectors_hit))
    support_fraction = count / max(1, len(normals))
    prior_angle = math.degrees(float(np.arccos(np.clip(best @ prior, -1., 1.))))
    inlier_degrees = np.degrees(residual[mask])
    if count:
        inlier_weights = np.array([weights[mask & (sectors == sector)].sum() for sector in sectors_hit])
        inlier_weights = np.minimum(inlier_weights, sector_cap)
        w = np.zeros(len(normals), dtype=float)
        for sector, sector_weight in zip(sectors_hit, inlier_weights):
            idx = mask & (sectors == sector)
            w[idx] = weights[idx] * min(1., sector_cap / max(weights[idx].sum(), 1e-12))
        eigenvalues = np.linalg.eigvalsh((normals * w[:, None]).T @ normals)
        eigen_gap = float((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1e-12))
    else:
        eigen_gap = 0.
    # A distinct alternative within the angular prior is evidence of a scene with no unique vertical direction.
    ambiguous = score_margin < float(min_score_margin)
    quality_ok = (support_fraction >= float(min_inlier_fraction)
                  and (float(np.median(inlier_degrees)) if count else np.inf) <= float(inlier_tolerance_deg) * .65
                  and eigen_gap >= float(min_eigen_gap) and not ambiguous)
    accepted = (count >= int(min_support) and sector_count >= int(min_sectors)
                and prior_angle <= float(max_prior_angle_deg) and quality_ok)
    holdout_support = holdout_sectors = None
    holdout_fraction = holdout_median = None
    train_agreement = None
    if groups is not None and _validate_holdout and len(np.unique(groups)) >= 4:
        train_mask = (groups % 2) == 0
        held_mask = ~train_mask
        if train_mask.sum() >= 12 and held_mask.sum() >= 8:
            train = _fit_gravity(normals[train_mask], weights[train_mask], sectors[train_mask], prior=prior,
                                 max_prior_angle_deg=max_prior_angle_deg, inlier_tolerance_deg=inlier_tolerance_deg,
                                 min_support=max(8, min_support // 2), min_sectors=2, min_inlier_fraction=.08,
                                 min_score_margin=.03, min_eigen_gap=min_eigen_gap * .5,
                                 ransac_iterations=ransac_iterations, _validate_holdout=False)
            if train['gravity'] is not None:
                held_residual = np.arcsin(np.clip(np.abs(normals[held_mask] @ train['gravity']), 0., 1.))
                held_inliers = held_residual <= math.radians(float(inlier_tolerance_deg))
                holdout_support = int(held_inliers.sum())
                holdout_sectors = int(len(np.unique(sectors[held_mask][held_inliers])))
                holdout_fraction = holdout_support / max(1, int(held_mask.sum()))
                holdout_median = float(np.degrees(np.median(held_residual[held_inliers]))) if holdout_support else None
                train_agreement = float(np.degrees(np.arccos(np.clip(train['gravity'] @ best, -1., 1.))))
                holdout_ok = (train['accepted'] and holdout_support >= max(6, min_support // 4)
                              and holdout_sectors >= 2 and holdout_fraction >= .08
                              and (holdout_median or np.inf) <= float(inlier_tolerance_deg) * .7
                              and train_agreement <= 2.5)
                accepted = accepted and holdout_ok
            else:
                accepted = False
        else:
            accepted = False

    if count < int(min_support):
        reason = 'insufficient consensus segments'
    elif sector_count < int(min_sectors):
        reason = 'consensus is confined to too few azimuth sectors'
    elif prior_angle > float(max_prior_angle_deg):
        reason = 'estimate exceeds prior angular bound'
    elif support_fraction < float(min_inlier_fraction):
        reason = 'consensus fraction is too small for non-random line evidence'
    elif ambiguous:
        reason = 'distinct gravity hypotheses have similar support'
    elif eigen_gap < float(min_eigen_gap):
        reason = 'line-plane normals do not define a unique gravity direction'
    elif count and float(np.median(inlier_degrees)) > float(inlier_tolerance_deg) * .65:
        reason = 'median angular residual is too large'
    elif groups is not None and _validate_holdout and not accepted:
        reason = 'consensus does not reproduce on held-out panorama views'
    else:
        reason = 'supported by distributed line evidence'
    return {
        'gravity': best.tolist(),
        'accepted': bool(accepted),
        'segment_count': int(len(normals)),
        'support': count,
        'support_sectors': sector_count,
        'support_fraction': float(support_fraction),
        'capped_support_score': float(best_score[0]),
        'runnerup_score_margin': float(score_margin),
        'eigen_gap': float(eigen_gap),
        'prior_angle_deg': prior_angle,
        'holdout_support': holdout_support,
        'holdout_sectors': holdout_sectors,
        'holdout_fraction': holdout_fraction,
        'holdout_median_residual_deg': holdout_median,
        'train_full_angle_deg': train_agreement,
        'angular_residuals_deg': inlier_degrees.tolist(),
        'median_residual_deg': float(np.median(inlier_degrees)) if count else None,
        'p90_residual_deg': float(np.percentile(inlier_degrees, 90)) if count else None,
        'reason': reason,
    }


def _load_gray(image):
    import cv2

    if isinstance(image, (str, Path)):
        source = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"could not read panorama: {image}")
        return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source = np.asarray(image)
    if source.ndim == 2:
        return _as_uint8(source)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("panorama must be a grayscale, RGB, or RGBA image")
    source = _as_uint8(source[..., :3])
    return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)


def _as_uint8(source):
    source = np.asarray(source)
    if source.dtype == np.uint8:
        return source
    source = source.astype(float)
    if not np.isfinite(source).all():
        raise ValueError("panorama pixels must be finite")
    if source.size and source.max() <= 1.:
        source *= 255.
    return np.clip(source, 0., 255.).astype(np.uint8)


def _view_maps(width, height, size, fov_deg, yaw_deg, pitch_deg):
    """Perspective pixel-to-pano and pixel-to-ray maps for a camera-coordinate view."""
    import cv2

    fov = math.radians(float(fov_deg))
    focal = size / (2. * math.tan(fov / 2.))
    xx, yy = np.meshgrid(np.arange(size, dtype=float) + .5 - size / 2.,
                         np.arange(size, dtype=float) + .5 - size / 2.)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([math.cos(pitch) * math.sin(yaw), -math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    right = np.array([math.cos(yaw), 0., -math.sin(yaw)])
    up = np.cross(right, forward)
    rays = forward + (xx / focal)[..., None] * right + (yy / focal)[..., None] * -up
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(-rays[..., 1], -1., 1.))
    map_x = np.mod((longitude / (2. * math.pi) + .5) * width, width).astype(np.float32)
    map_y = np.clip((.5 - latitude / math.pi) * height, 0., height - 1.).astype(np.float32)
    return map_x, map_y, rays


def estimate_panorama_gravity(image, *, prior=_NOMINAL_UP, view_size=500, fov_deg=80.,
                              azimuth_step_deg=45., upward_pitch_deg=22., min_line_deg=3.,
                              inlier_tolerance_deg=2.5, max_prior_angle_deg=35.,
                              min_support=20, min_sectors=3):
    """Estimate camera-space world-up from an equirectangular panorama.

    Eight equatorial views cover the full azimuth; four additional upward-looking views capture upper wall courses.
    The return dictionary includes the unit ``gravity``/world-up vector, angular residuals, and acceptance evidence.
    """
    import cv2

    gray = _load_gray(image)
    height, width = gray.shape[:2]
    if width < height * 1.5:
        raise ValueError("expected an equirectangular panorama wider than 1.5:1")
    step = float(azimuth_step_deg)
    if not np.isfinite(step) or step <= 0 or 360. % step > 1e-6:
        raise ValueError("azimuth_step_deg must divide 360")
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    observations = []
    equatorial = np.arange(0., 360., step)
    views = [(float(yaw), 0.) for yaw in equatorial]
    views.extend((float(yaw), float(upward_pitch_deg)) for yaw in equatorial[::max(1, int(round(90. / step)))])
    sector_count = max(4, int(round(360. / step)))

    for view_index, (yaw_deg, pitch_deg) in enumerate(views):
        mx, my, rays = _view_maps(width, height, int(view_size), fov_deg, yaw_deg, pitch_deg)
        view = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        detected = detector.detect(view)[0]
        if detected is None:
            continue
        for raw in np.asarray(detected).reshape((-1, 4)):
            x0, y0, x1, y1 = map(float, raw)
            if math.hypot(x1 - x0, y1 - y0) < 14.:
                continue
            # Sample the perspective ray field at the detected pixel endpoints.
            a = np.array([np.interp(x0, np.arange(view_size), rays[int(np.clip(round(y0), 0, view_size - 1))].T[k])
                          for k in range(3)])
            b = np.array([np.interp(x1, np.arange(view_size), rays[int(np.clip(round(y1), 0, view_size - 1))].T[k])
                          for k in range(3)])
            middle = _unit(a + b)
            azimuth = math.atan2(float(middle[0]), float(middle[2])) % (2. * math.pi)
            sector = int(azimuth / (2. * math.pi) * sector_count) % sector_count
            observation = _line_observation(a, b, sector, min_line_deg)
            if observation is not None:
                observations.append((*observation, view_index))

    if not observations:
        return {'gravity': None, 'accepted': False, 'segment_count': 0, 'support': 0,
                'support_sectors': 0, 'angular_residuals_deg': [], 'reason': 'no usable line segments'}
    normals, weights, sectors, groups = map(np.asarray, zip(*observations))
    return _fit_gravity(normals, weights, sectors, prior=prior,
                        max_prior_angle_deg=max_prior_angle_deg,
                        inlier_tolerance_deg=inlier_tolerance_deg,
                        min_support=min_support, min_sectors=min_sectors, groups=groups)


def corrected_rotation(rotation, gravity):
    """Apply the minimal proper rotation that maps ``rotation @ [0,0,1]`` onto camera-space ``gravity``."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    source, target = _unit(rotation @ np.array([0., 0., 1.])), _unit(gravity)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1., 1.))
    if sine < 1e-12:
        if cosine > 0:
            return rotation.copy()
        # The antipodal case has no unique minimal heading-preserving axis.
        raise ValueError("cannot choose a unique correction for antipodal up vectors")
    skew = np.array([[0., -cross[2], cross[1]],
                     [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    correction = np.eye(3) + skew + (skew @ skew) * ((1. - cosine) / (sine * sine))
    result = correction @ rotation
    return result
