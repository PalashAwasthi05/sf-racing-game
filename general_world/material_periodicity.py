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
            classes = [min(abs(x), abs(x - pitch), abs(x - half)) for x in offsets]
            separated = sum(abs(x - half) <= max(2, pitch * .18) for x in offsets) >= 2
            banded["stagger_consistent"] = separated and len(observed) >= 4
            if banded["stagger_consistent"]:
                banded["full_module_period_px"] = int(round(pitch))
                banded["accepted"] = True
                banded["ambiguous_half_period"] = False
        if len(observed) >= 4:
            short = float(np.median([x for x in observed if x <= np.median(observed)]))
            doubled = [x for x in observed if abs(x / max(short, 1) - 2.0) <= 0.25]
            short_count = sum(abs(x / max(short, 1) - 1.0) <= 0.25 for x in observed)
            # Alternating short/full pitches are the measurable signature of a
            # staggered bond; the physical brick width remains unclaimed here.
            if short_count >= 2 and len(doubled) >= 2:
                full = float(np.median(doubled))
                banded["full_module_period_px"] = int(round(full))
                banded["stagger_consistent"] = True
                banded["accepted"] = True
                banded["ambiguous_half_period"] = False
    accepted = horizontal["period_px"] is not None or vertical["period_px"] is not None
    return {"accepted": accepted or banded["accepted"], "horizontal": horizontal, "vertical": vertical,
            "banded_running_bond": banded,
            "signal": "row/column median edge profile", "physical_scale": "not inferred"}
