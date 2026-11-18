"""Measure repeated panel joints in masked facade photographs.

Unlike a texture-map periodicity measurement, this tolerates a sheared image,
windows and partial occlusion. Image coordinates remain image coordinates:
neither camera registration nor metric dimensions are certified here.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


def _project(ridge, mask, shear, start, stop, fan=0.):
    """Project ridges through a bounded one-vanishing-point correction."""
    h, w = ridge.shape
    y, x = np.nonzero(mask[start:stop])
    y = y + start
    # Padding keeps all sheared coordinates in bounds. No cyclic wrapping.
    pad = int(np.ceil(h * .4)) + 2
    q = np.rint((x - shear * y) / (1 + fan * y / h)).astype(int) + pad
    n = int(np.ceil(w * 1.5)) + 2 * pad
    weights = np.bincount(q, minlength=n).astype(float)
    total = np.bincount(q, weights=ridge[y, x], minlength=n)
    profile = np.divide(total, weights, out=np.zeros(n), where=weights > 0)
    supported = weights >= max(4, (stop - start) * .08)
    profile[~supported] = 0
    return gaussian_filter1d(profile, .65), supported, pad


def _fit(profile, supported, pad, minimum, maximum):
    active = profile[supported]
    if active.size < minimum * 4 or float(np.max(active, initial=0)) < 1e-5:
        return None
    floor = float(np.median(active))
    mad = float(np.median(np.abs(active - floor)))
    threshold = floor + max(2.5 * mad, float(active.max()) * .10, .001)
    peaks, props = find_peaks(profile, height=threshold,
                             prominence=max(mad * 2, .001), distance=max(2, minimum // 3))
    peaks = peaks[supported[peaks]]
    if len(peaks) < 5:
        return None
    strengths = profile[peaks]
    best = None
    # Distances between observed joints provide candidates; missing joints are
    # allowed, but no lattice earns support from an interpolated line.
    distances = peaks[:, None] - peaks[None, :]
    pitches = sorted(set(float(d) for d in distances.ravel() if minimum <= d <= maximum))
    for pitch in pitches:
        for anchor in peaks[np.argsort(strengths)[-min(8, len(peaks)):]]:
            phase = float(anchor % pitch)
            residual = np.abs((peaks - phase + pitch / 2) % pitch - pitch / 2)
            inlier = residual <= max(1.5, pitch * .075)
            if np.count_nonzero(inlier) < 5:
                continue
            # Refit exact line positions, keeping the observed integer indices.
            indices = np.rint((peaks[inlier] - phase) / pitch)
            if len(np.unique(indices)) != len(indices):
                continue
            fitted_pitch, intercept = np.polyfit(indices, peaks[inlier], 1)
            if not minimum <= fitted_pitch <= maximum:
                continue
            err = np.abs(peaks[inlier] - (indices * fitted_pitch + intercept))
            if np.max(err) > max(2., fitted_pitch * .09):
                continue
            indices_span = int(indices.max() - indices.min() + 1)
            coverage = len(indices) / indices_span
            if coverage < .65 or indices_span < 5:
                continue
            energy = float(strengths[inlier].sum() / max(strengths.sum(), 1e-9))
            # An isolated bright edge must not outweigh five weak real joints.
            score = energy * coverage * (1 - min(.5, float(np.mean(err)) / fitted_pitch))
            candidate = dict(pitch_px=float(fitted_pitch),
                             phase_px=float((intercept - pad) % fitted_pitch),
                             observed_lines=int(len(indices)), line_coverage=float(coverage),
                             residual_px=float(np.sqrt(np.mean(err ** 2))), score=score)
            if best is None or score > best['score']:
                best = candidate
    return best


def _measure_axis(ridge, mask, minimum, maximum, shear_limit):
    h, _ = ridge.shape
    best = None
    # Interleaved independent strips: fit on even strips, check odd strips.
    # Separate rows prevent the same pixels from validating their own fit.
    strip_id = np.minimum(7, np.arange(h) * 8 // h)
    train = mask & ((strip_id % 2 == 0)[:, None])
    def search(shears, fans):
        nonlocal best
        for fan in fans:
            for shear in shears:
                p, valid, pad = _project(ridge, train, float(shear), 0, h, float(fan))
                fit = _fit(p, valid, pad, minimum, maximum)
                if fit is None:
                    continue
                # Prefer sharp line alignment, not only a recurring broad shadow.
                score = fit['score'] * float(np.percentile(p[valid], 98))
                if best is None or score > best['selection_score']:
                    best = {**fit, 'shear': float(shear), 'fan': float(fan), 'selection_score': score}
    search(np.linspace(-shear_limit, shear_limit, 31), np.linspace(-.24, .24, 9))
    if best is not None:
        search(np.linspace(max(-shear_limit, best['shear']-.015),
                           min(shear_limit, best['shear']+.015), 9),
               np.linspace(max(-.28, best['fan']-.04), min(.28, best['fan']+.04), 9))
    if best is None:
        return {'accepted': False, 'reason': 'no supported repeat with five observed lines'}
    pitch, phase, shear, fan = best['pitch_px'], best['phase_px'], best['shear'], best['fan']
    checks = []
    for strip in range(1, 8, 2):
        start, stop = strip * h // 8, (strip + 1) * h // 8
        p, valid, pad = _project(ridge, mask, shear, start, stop, fan)
        active = p[valid]
        if len(active) < minimum * 3 or float(np.max(active, initial=0)) < .002:
            continue
        peaks, _ = find_peaks(p, prominence=max(.001, float(active.max()) * .1),
                              distance=max(2, minimum // 3))
        peaks = peaks[valid[peaks]]
        # Test predicted lines, including absent ones, rather than letting
        # arbitrary clutter peaks vote for the proposed grid.
        lines = np.arange(np.floor((-pad - phase) / pitch),
                          np.ceil((len(p) - pad - phase) / pitch) + 1) * pitch + phase + pad
        lines = [q for q in lines if 1 <= q < len(p) - 1 and valid[int(round(q))]]
        if len(lines) < 4:
            continue
        errors = [float(np.min(np.abs(peaks - q))) if len(peaks) else float(pitch) for q in lines]
        fraction = float(np.mean(np.array(errors) <= max(2, pitch * .10)))
        near = np.abs((peaks - pad - phase + pitch/2) % pitch - pitch/2) <= max(2, pitch*.10)
        # A dense field of random grain has a nearby peak at almost every
        # prediction. Require the repeat to explain the observed ridge energy,
        # not just find a convenient subset of clutter.
        explained = float(p[peaks[near]].sum() / max(float(p[peaks].sum()), 1e-9))
        checks.append({'strip': strip, 'predicted_visible_lines': len(lines),
                       'supported_fraction': round(fraction, 4),
                       'explained_ridge_energy': round(explained, 4),
                       'median_error_px': round(float(np.median(errors)), 3)})
    passed = sum(c['supported_fraction'] >= .65 and c['explained_ridge_energy'] >= .55 for c in checks)
    accepted = passed >= 2 and passed >= len(checks) * .6
    best.pop('selection_score')
    return {**{k: round(v, 5) if isinstance(v, float) else v for k, v in best.items()},
            'accepted': accepted, 'holdout_strips': checks,
            'reason': 'repeat supported in independent image strips' if accepted
                      else 'repeat failed independent image-strip checks'}


def measure_joint_grid(image, surface_mask, *, min_pitch_px=12, max_pitch_px=180,
                       shear_limit=.20):
    """Return image-space joint candidates, separately gated on each axis.

    surface_mask selects ONE material zone and excludes openings/occluders.
    It is an input observation, not an inferred substrate classification.
    At most 1536px per image dimension bounds search and temporary memory.
    """
    rgb = np.asarray(image, dtype=float)
    mask = np.asarray(surface_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('RGB image and same-size surface mask required')
    if min(mask.shape) < 64 or max(mask.shape) > 1536 or not np.isfinite(rgb).all():
        raise ValueError('finite image, dimensions from 64 through 1536 required')
    if (rgb < 0).any() or (rgb > 255).any():
        raise ValueError('RGB values must be in [0,255]')
    if not 4 <= min_pitch_px < max_pitch_px <= max(mask.shape) // 2:
        raise ValueError('invalid pitch interval for image size')
    if not np.isfinite(shear_limit) or not 0 <= shear_limit <= .24:
        raise ValueError('shear limit must be in [0,.24]')
    gray = rgb.mean(axis=2) / 255
    # Erode window/foliage boundaries so their high contrast cannot create
    # fictitious material joints. Normalize local blur by surface coverage.
    mask = binary_erosion(mask, iterations=3, border_value=0)
    weights = gaussian_filter(mask.astype(float), 3)
    smooth = gaussian_filter(gray * mask, 3) / np.maximum(weights, 1e-8)
    ridge = np.maximum(0, smooth - gray)
    ridge[~mask] = 0
    vertical = _measure_axis(ridge, mask, min_pitch_px, min(max_pitch_px, mask.shape[1]//2), shear_limit)
    horizontal = _measure_axis(ridge.T, mask.T, min_pitch_px, min(max_pitch_px, mask.shape[0]//2), shear_limit)
    return {'schema': 'facade-joint-grid/1', 'vertical': vertical, 'horizontal': horizontal,
            'coordinate_frame': 'input image pixels; q=(axis-shear*other_axis)/(1+fan*other_axis/other_axis_extent)',
            'metric_scale_verified': False, 'world_registration_verified': False,
            'scope': 'shallow joint repetition only; not facade layout or material identity'}
