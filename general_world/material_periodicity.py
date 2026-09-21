"""Small, evidence-only periodicity estimator for scanned material maps.

The estimator is intentionally conservative: it reports pixel periods in the
input map and rejects signals without repeated, joint-like structure.  It does
not attach physical units or infer a material family.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _profile(image: np.ndarray, axis: int) -> np.ndarray:
    """Return a robust one-dimensional edge profile along ``axis``."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2 or min(arr.shape) < 8:
        raise ValueError("image must be a 2-D array with at least 8 pixels per side")
    # A joint is represented by a concentrated change in the row/column mean.
    values = np.nanmedian(arr, axis=1 if axis == 0 else 0)
    values = values - np.polyval(np.polyfit(np.arange(values.size), values, 1), np.arange(values.size))
    delta = np.diff(values, prepend=values[0])
    scale = np.median(np.abs(delta - np.median(delta)))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(delta))
    return delta / max(scale, 1e-9)


def _one(profile: np.ndarray, *, min_period_px: int, max_period_px: int) -> dict[str, Any]:
    n = profile.size
    if max_period_px > n // 2:
        max_period_px = n // 2
    if max_period_px < min_period_px or np.std(profile) < 1e-6:
        return {"period_px": None, "confidence": 0.0, "uncertainty_px": None, "cycles": 0.0}
    centered = profile - profile.mean()
    ac = np.correlate(centered, centered, mode="full")[n - 1 :]
    ac = ac / max(float(ac[0]), 1e-9)
    candidates = []
    for lag in range(min_period_px, max_period_px + 1):
        if lag == max_period_px or (ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]):
            candidates.append(lag)
    if not candidates:
        return {"period_px": None, "confidence": 0.0, "uncertainty_px": None, "cycles": 0.0}
    lag = max(candidates, key=lambda x: float(ac[x]))
    peak = float(ac[lag])
    nearby = [float(ac[x]) for x in candidates if abs(x - lag) <= max(2, lag // 8)]
    baseline = float(np.percentile(ac[min_period_px : max_period_px + 1], 50))
    prominence = peak - baseline
    # Concentrated edge changes distinguish joints from low-frequency grain.
    sharpness = float(np.percentile(np.abs(profile), 90) / max(np.median(np.abs(profile)), 1e-6))
    confidence = float(np.clip(0.55 * max(0.0, peak) + 0.35 * prominence + 0.10 * min(sharpness / 4.0, 1.0), 0, 1))
    accepted = peak >= 0.22 and prominence >= 0.10 and sharpness >= 1.8 and n / lag >= 2.5
    if not accepted:
        return {"period_px": None, "confidence": round(confidence, 4), "uncertainty_px": None, "cycles": round(n / lag, 3)}
    spread = max(1, int(round(lag * 0.08)))
    return {"period_px": int(lag), "confidence": round(confidence, 4), "uncertainty_px": spread, "cycles": round(n / lag, 3)}


def infer_joint_periods(image: np.ndarray, *, min_period_px: int = 4,
                        max_period_px: int | None = None) -> dict[str, Any]:
    """Infer repeated horizontal and vertical joint pitch from map evidence.

    ``horizontal`` is the row-to-row pitch and ``vertical`` is the column-to-
    column pitch.  The result is suitable for a measured ``unit_size_m``
    conversion only after an external physical scale is supplied.
    """
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must be a 2-D or 3-D array")
    if not np.isfinite(arr).all():
        raise ValueError("image contains non-finite samples")
    h, w = arr.shape[:2]
    cap = max_period_px or min(h, w) // 2
    horizontal = _one(_profile(arr, 0), min_period_px=min_period_px, max_period_px=min(cap, h // 2))
    vertical = _one(_profile(arr, 1), min_period_px=min_period_px, max_period_px=min(cap, w // 2))
    banded = {"accepted": False, "row_pitch_px": horizontal["period_px"],
              "band_periods_px": [], "full_module_period_px": None,
              "stagger_consistent": False, "ambiguous_half_period": True}
    row_pitch = horizontal["period_px"]
    if row_pitch is not None and row_pitch >= min_period_px * 2:
        profiles = []
        raw_profiles = []
        for start in range(0, h - row_pitch + 1, row_pitch):
            band = arr[start:start + row_pitch]
            if band.ndim == 3:
                band = band.mean(axis=2)
            values = np.nanmedian(band, axis=0)
            raw = np.diff(values, prepend=values[0]); raw_profiles.append(raw - raw.mean())
            profiles.append(_one(np.diff(values, prepend=values[0]),
                                 min_period_px=min_period_px,
                                 max_period_px=min(cap, w // 2)))
        observed = [x["period_px"] for x in profiles if x["period_px"] is not None]
        banded["band_periods_px"] = observed
        if len(raw_profiles) >= 4 and observed:
            pitch = int(round(float(np.median(observed))))
            ref = raw_profiles[0] / max(float(np.linalg.norm(raw_profiles[0])), 1e-9)
            offsets = []
            for raw in raw_profiles:
                raw = raw / max(float(np.linalg.norm(raw)), 1e-9)
                shifts = range(-pitch // 2, pitch // 2 + 1)
                best = max(shifts, key=lambda s: float(np.dot(ref, np.roll(raw, s))))
                offsets.append(int(best % pitch))
            banded["phase_offsets_px"] = offsets
            # Two stable phase classes separated by half a pitch certify a
            # staggered repeat; a single class remains ambiguous.
            half = pitch / 2.0
            tolerance = max(2, pitch * .12)
            labels = [1 if abs(x - half) <= tolerance else
                      (0 if min(abs(x), abs(x - pitch)) <= tolerance else -1)
                      for x in offsets]
            valid = [x for x in labels if x >= 0]
            alternations = sum(a != b for a, b in zip(labels, labels[1:]) if a >= 0 and b >= 0)
            comparisons = sum(a >= 0 and b >= 0 for a, b in zip(labels, labels[1:]))
            phase_fraction = sum(x >= 0 for x in labels) / max(len(labels), 1)
            alternate_fraction = alternations / max(comparisons, 1)
            separated = sum(x == 1 for x in valid) >= 2 and sum(x == 0 for x in valid) >= 2
            banded["phase_consistency"] = round(min(phase_fraction, alternate_fraction), 4)
            banded["stagger_consistent"] = (separated and len(observed) >= 4 and
                                             phase_fraction >= .75 and alternate_fraction >= .75)
            if banded["stagger_consistent"]:
                banded["full_module_period_px"] = int(round(pitch))
                banded["accepted"] = True
                banded["ambiguous_half_period"] = False
    accepted = horizontal["period_px"] is not None or vertical["period_px"] is not None
    return {"accepted": accepted or banded["accepted"], "horizontal": horizontal, "vertical": vertical,
            "banded_running_bond": banded,
            "signal": "row/column median edge profile", "physical_scale": "not inferred"}


def _course_phase(arr: np.ndarray, row_px: int) -> int:
    """Row offset of the horizontal joints, so that bands read one course each rather than straddling two."""
    means = np.nanmedian(arr, axis=1)
    edge = np.abs(np.diff(means, prepend=means[0]))
    h = arr.shape[0]
    scores = [sum(edge[(phase + k * row_px) % h] for k in range(h // row_px)) for phase in range(row_px)]
    return int(np.argmax(scores))


def _course_profiles(arr: np.ndarray, row_px: int) -> list[np.ndarray]:
    """Vertical-joint profile of each course: the column derivative of the course's median row, joint rows excluded."""
    h = arr.shape[0]
    phase = _course_phase(arr, row_px)
    margin = max(1, row_px // 8)
    out = []
    for start in range(phase, h - row_px + 1, row_px):
        band = arr[start + margin:start + row_px - margin]
        if band.shape[0] < 2:
            continue
        values = np.nanmedian(band, axis=0)
        d = np.diff(values, prepend=values[0])
        out.append(d - d.mean())
    return out


def measure_masonry_module(image: np.ndarray, *, min_period_px: int = 6) -> dict[str, Any]:
    """Course height and full unit module in pixels, and the bond, from one map alone.

    The course pitch is the row period; each course is then read on its own (aligned to the joint rows, so
    a band never mixes two courses and the half-module ambiguity of a running bond does not arise); the module
    is the joint period shared by the courses; the bond is the shift between neighbouring courses (half a module
    for a running bond, none for a stack bond). No physical unit is attached and no material family inferred.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2 or min(arr.shape) < 32:
        return {"accepted": False, "reason": "map too small"}
    h, w = arr.shape
    rows = _one(_profile(arr, 0), min_period_px=min_period_px, max_period_px=h // 2)
    row_px = rows["period_px"]
    if not row_px:
        return {"accepted": False, "reason": "no course pitch"}
    courses = _course_profiles(arr, row_px)
    if len(courses) < 4:
        return {"accepted": False, "reason": "too few courses"}
    # Mean autocorrelation of the normalised course profiles: the smallest strong peak is the module.
    acc = np.zeros(w)
    for p in courses:
        c = p - p.mean()
        ac = np.correlate(c, c, "full")[w - 1:]
        acc += ac / max(float(ac[0]), 1e-9)
    acc /= len(courses)
    lags = [k for k in range(min_period_px, w // 2) if acc[k] >= acc[k - 1] and acc[k] >= acc[k + 1] and acc[k] > .15]
    if not lags:
        return {"accepted": False, "reason": "no joint period across the courses"}
    top = max(float(acc[k]) for k in lags)
    module_px = min(k for k in lags if acc[k] >= .8 * top)
    # Bond: the circular shift that best aligns neighbouring courses.
    shifts = []
    for a, b in zip(courses, courses[1:]):
        na = a / max(float(np.linalg.norm(a)), 1e-9)
        nb = b / max(float(np.linalg.norm(b)), 1e-9)
        shifts.append(max(range(module_px), key=lambda s: float(np.dot(na, np.roll(nb, s)))))
    half = module_px / 2
    tol = max(2, .15 * module_px)
    running = sum(abs(s - half) <= tol for s in shifts) / len(shifts)
    stacked = sum(min(s, module_px - s) <= tol for s in shifts) / len(shifts)
    # Hand-laid bricks vary in length by a tenth or more, so the course-to-course shift is only a diagnostic; the
    # module itself is certified by the course pitch and the joint period read inside single courses.
    bond = "running" if running >= .7 else "stack" if stacked >= .7 else None
    return {"accepted": True, "bond": bond, "row_px": int(row_px), "module_px": int(module_px), "peak": round(top, 3),
            "shifts_px": shifts, "stagger_fraction": round(max(running, stacked), 3), "courses": len(courses),
            "physical_scale": "not inferred"}


def measure_board_pitch(image: np.ndarray, *, min_period_px: int = 8) -> dict[str, Any]:
    """Pitch of parallel boards in pixels and the axis they run along, from one map alone."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape
    rows = _one(_profile(arr, 0), min_period_px=min_period_px, max_period_px=h // 2)
    cols = _one(_profile(arr, 1), min_period_px=min_period_px, max_period_px=w // 2)
    best, other, axis = (rows, cols, "rows") if rows["confidence"] >= cols["confidence"] else (cols, rows, "columns")
    if not best["period_px"] or best["confidence"] < .4:
        return {"accepted": False, "reason": "no board pitch"}
    if other["period_px"] and other["confidence"] > .8 * best["confidence"]:
        return {"accepted": False, "reason": "joints in both directions are not boards"}
    # boards run along the axis perpendicular to the pitch: a row pitch means horizontal boards
    return {"accepted": True, "pitch_px": int(best["period_px"]), "boards": "horizontal" if axis == "rows" else "vertical",
            "confidence": best["confidence"], "physical_scale": "not inferred"}
