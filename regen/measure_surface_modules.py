"""Conservative module proposals from independently masked rectified wall views.

This only measures periodic pixel evidence. It does not infer a surface type,
attach a module to a profile, or use renderer output. Masks are the existing
appearance-patch texture masks (openings/projections removed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MIN_PERIOD_PX = 6  # minimum reliable repeat after rectification/antialiasing
STRIPS = 4


def _gray(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim == 2:
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("expected grayscale or RGB image")
    return rgb[..., :3] @ np.array([0.2126, 0.7152, 0.0722])


def _line_profile(gray: np.ndarray, mask: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    # axis 0: one value per image row; axis 1: one value per column.
    reduce_axis = 1 if axis == 0 else 0
    weights = np.asarray(mask, dtype=bool).sum(axis=reduce_axis).astype(float)
    total = mask.shape[reduce_axis]
    values = np.full(weights.shape, np.nan, dtype=float)
    for i, count in enumerate(weights.astype(int)):
        if count:
            row = gray[i, mask[i]] if axis == 0 else gray[mask[:, i], i]
            values[i] = float(np.median(row))
    # Missing line samples are acceptable only as isolated short gaps. Large
    # occlusions would let interpolation invent periodic structure.
    valid = np.isfinite(values) & (weights >= max(4, .12 * total))
    if valid.mean() < .45:
        return values, valid
    ids = np.flatnonzero(valid)
    gaps = np.diff(ids) - 1
    # Interpolate short holes for display/fit only. Long masked regions remain
    # invalid and are excluded from the lag correlation below.
    if len(gaps) and int(gaps.max()) <= max(3, int(.08 * len(values))):
        values[~valid] = np.interp(np.flatnonzero(~valid), ids, values[valid])
    return values, valid


def _period(profile: np.ndarray, valid: np.ndarray, *, min_period_px: int = MIN_PERIOD_PX) -> dict[str, Any]:
    n = len(profile)
    if n < 48 or valid.mean() < .45:
        return {"accepted": False, "reason": "insufficient contiguous profile"}
    ids = np.flatnonzero(valid & np.isfinite(profile))
    if len(ids) < 32:
        return {"accepted": False, "reason": "too few valid line samples"}
    # Remove illumination ramp, then emphasize seams/edges rather than flat color.
    fit = np.polyfit(ids, profile[ids], 1)
    filled = np.interp(np.arange(n), ids, profile[ids])
    raw = np.diff(filled)
    raw = raw - raw.mean()
    raw_energy = float(raw @ raw)
    if raw_energy > 1e-5:
        undersampled = []
        for lag in range(2, min_period_px):
            left = np.arange(0, len(raw) - lag)
            good = valid[left] & valid[left + 1] & valid[left + lag] & valid[left + lag + 1]
            if int(good.sum()) >= max(24, int(.12 * (len(raw) - lag))):
                a, b = raw[left[good]], raw[left[good] + lag]
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                if den > 1e-9:
                    undersampled.append((float(a @ b / den), lag))
        if undersampled and max(undersampled)[0] >= .65:
            score, lag = max(undersampled)
            return {"accepted": False, "period_px": None, "belowNyquist": True,
                    "subpixelAliasPx": lag, "aliasCorrelation": round(score, 4),
                    "reason": "strong repeat below the six-pixel reliability threshold"}
    smooth = np.convolve(filled, np.ones(9) / 9, mode="same")
    signal = np.diff(filled - smooth)
    sig_valid = valid[1:] & valid[:-1]
    signal -= float(np.mean(signal[sig_valid]))
    energy = float(signal[sig_valid] @ signal[sig_valid])
    if energy < 1e-5:
        return {"accepted": False, "reason": "no repeat energy"}
    ac = np.full(len(signal), np.nan, dtype=float)
    ac[0] = 1.
    for lag in range(1, len(signal)):
        left = np.arange(0, len(signal) - lag)
        good = sig_valid[left] & sig_valid[left + lag]
        if int(good.sum()) < max(24, int(.12 * (n - lag))):
            continue
        x, y = signal[left[good]], signal[left[good] + lag]
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        if den > 1e-9:
            ac[lag] = float(x @ y / den)
    low = max(2, min_period_px)
    high = n // 2
    peaks = [k for k in range(low + 1, high) if np.isfinite(ac[k-1:k+2]).all() and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return {"accepted": False, "reason": "no candidate repeat"}
    # Prefer the first strong periodic peak, not a later multiple of the pitch.
    max_peak = max(float(ac[k]) for k in peaks)
    candidate = min((k for k in peaks if ac[k] >= .8 * max_peak), default=None)
    if candidate is None:
        return {"accepted": False, "reason": "weak repeat"}
    baseline = float(np.nanmedian(ac[low:high]))
    peak = float(ac[candidate])
    prominence = peak - baseline
    cycles = n / candidate
    accepted = peak >= .22 and prominence >= .10 and cycles >= 2.5
    if not accepted:
        return {"accepted": False, "period_px": int(candidate), "peak": round(peak, 4),
                "prominence": round(prominence, 4), "cycles": round(cycles, 2),
                "reason": "weak peak or too few cycles"}
    # Parabolic interpolation around the discrete peak (bounded to half a pixel).
    a, b, c = float(ac[candidate - 1]), peak, float(ac[candidate + 1])
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.
    period = float(candidate + np.clip(delta, -.5, .5))
    return {"accepted": True, "period_px": round(period, 3), "peak": round(peak, 4),
            "prominence": round(prominence, 4), "cycles": round(cycles, 2)}


def measure_view(image: np.ndarray, mask: np.ndarray, ppm: float, *, min_coverage: float = .25) -> dict[str, Any]:
    """Measure horizontal and vertical repeats across four independent wall strips."""
    gray = _gray(image)
    valid = np.asarray(mask, dtype=bool)
    if gray.shape != valid.shape:
        raise ValueError("image and valid mask dimensions differ")
    if not np.isfinite(gray).all() or ppm <= 0:
        raise ValueError("image values and pixels-per-metre must be finite and positive")
    coverage = float(valid.mean())
    result: dict[str, Any] = {"accepted": False, "ppm": float(ppm), "coverage": round(coverage, 5),
                              "minimum_period_px": MIN_PERIOD_PX, "axes": {}}
    if coverage < min_coverage:
        result["reason"] = "valid wall coverage below acceptance gate"
        return result
    h, w = valid.shape
    for axis, label, span in ((0, "vertical_repeat_px", h), (1, "horizontal_repeat_px", w)):
        strip_edges = np.linspace(0, w if axis == 0 else h, STRIPS + 1, dtype=int)
        estimates = []
        for i in range(STRIPS):
            lo, hi = strip_edges[i:i + 2]
            if axis == 0:
                sub_gray, sub_mask = gray[:, lo:hi], valid[:, lo:hi]
            else:
                sub_gray, sub_mask = gray[lo:hi, :], valid[lo:hi, :]
            if sub_mask.size == 0 or sub_mask.mean() < .15:
                estimates.append({"accepted": False, "reason": "strip coverage too low"})
                continue
            profile, line_valid = _line_profile(sub_gray, sub_mask, axis)
            if line_valid.mean() < .45:
                estimates.append({"accepted": False, "reason": "mask fragments profile"})
                continue
            estimates.append(_period(profile, line_valid))
        good = [x for x in estimates if x.get("accepted")]
        if len(good) >= 3:
            periods = np.array([x["period_px"] for x in good], dtype=float)
            med = float(np.median(periods))
            dispersion = float(np.median(np.abs(periods - med)) / med)
            axis_accepted = dispersion <= .12
        else:
            med, dispersion, axis_accepted = None, None, False
        result["axes"][label] = {"accepted": axis_accepted,
                                  "period_px": round(med, 3) if med is not None and axis_accepted else None,
                                  "period_m": round(med / ppm, 5) if med is not None and axis_accepted else None,
                                  "stripAgreementMadFraction": round(dispersion, 4) if dispersion is not None else None,
                                  "acceptedStrips": len(good), "strips": estimates}
    result["accepted"] = any(v["accepted"] for v in result["axes"].values())
    result["physical_scale_status"] = "rectified pixels divided by source crop ppm"
    return result


def compare_views(a: dict[str, Any], b: dict[str, Any], *, agreement_fraction: float = .12) -> dict[str, Any]:
    """Accept a scale only when independently masked source views agree."""
    result: dict[str, Any] = {"accepted": False, "agreementFraction": agreement_fraction, "axes": {}}
    for axis in ("vertical_repeat_px", "horizontal_repeat_px"):
        x, y = a.get("axes", {}).get(axis, {}), b.get("axes", {}).get(axis, {})
        px, py = x.get("period_m"), y.get("period_m")
        accepted = bool(x.get("accepted") and y.get("accepted") and px and py and
                        abs(px - py) / max((px + py) / 2, 1e-9) <= agreement_fraction and
                        abs(a.get("ppm", 0) - b.get("ppm", 0)) <= .01 * max(a.get("ppm", 0), b.get("ppm", 0)))
        result["axes"][axis] = {"accepted": accepted,
                                  "period_m": round((px + py) / 2, 5) if accepted else None,
                                  "viewA_m": px, "viewB_m": py,
                                  "reason": None if accepted else "views lack an accepted repeat or disagree"}
    result["accepted"] = any(x["accepted"] for x in result["axes"].values())
    return result


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def run(report_path: Path, sides_paths: list[Path], patch_dir: Path, output_path: Path,
        *, min_coverage: float = .25) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    by_id = {}
    for sides_path in sides_paths:
        side_sets = json.loads(sides_path.read_text())
        by_id.update({piece["id"]: piece for piece in side_sets["pieces"]})
    records = []
    for rec in report.get("records", []):
        rid = rec["readingId"]
        piece = by_id.get(rid)
        if not piece or len(piece.get("views", [])) < 2:
            records.append({"readingId": rid, "accepted": False, "reason": "two calibrated rectified views unavailable"})
            continue
        stem = f"{rid}__{rec['family']}"
        img_a = _load_rgb(patch_dir / f"{stem}__source-a-raw.png")
        img_b = _load_rgb(patch_dir / f"{stem}__heldout-b-normalized.png")
        mask_a = np.asarray(Image.open(patch_dir / f"{stem}__valid-mask.png").convert("L")) > 0
        mask_b = np.asarray(Image.open(patch_dir / f"{stem}__heldout-valid-mask.png").convert("L")) > 0
        ppm_a, ppm_b = float(piece["views"][0]["ppm"]), float(piece["views"][1]["ppm"])
        ma = measure_view(img_a, mask_a, ppm_a, min_coverage=min_coverage)
        mb = measure_view(img_b, mask_b, ppm_b, min_coverage=min_coverage)
        result = compare_views(ma, mb)
        diagnostic_files = {}
        for label, suffix in (("sourceA", "source-a-raw.png"), ("maskA", "valid-mask.png"),
                              ("heldoutB", "heldout-b-normalized.png"), ("maskB", "heldout-valid-mask.png")):
            path = patch_dir / f"{stem}__{suffix}"
            diagnostic_files[label] = {"path": str(path),
                                       "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
        records.append({"readingId": rid, "family": rec.get("family"), "zone": rec.get("zone"),
                        "sourceMaskCoverage": [ma["coverage"], mb["coverage"]],
                        "imageIds": [v["imageId"] for v in rec["views"][:2]],
                        "diagnosticFiles": diagnostic_files,
                        "viewA": ma, "viewB": mb, "agreement": result,
                        "accepted": result["accepted"],
                        "limitations": ["no material family or pattern inferred", "no profile update", "normalized held-out RGB used only for periodicity"]})
    output = {"schema": "surface-module-proposals-v1", "sourceReport": str(report_path),
              "sourceReportSha256": __import__("hashlib").sha256(report_path.read_bytes()).hexdigest(),
              "sideSources": [str(p) for p in sides_paths],
              "patchDirectory": str(patch_dir), "minimumValidCoverage": min_coverage,
              "minimumRepeatPx": MIN_PERIOD_PX, "records": records,
              "acceptedCount": sum(bool(r.get("accepted")) for r in records),
              "appliedToProfiles": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="appearance-patches report.json")
    parser.add_argument("--sides", type=Path, action="append", required=True,
                        help="side profile JSON containing pixels-per-metre; repeat for multiple shards")
    parser.add_argument("--patch-dir", type=Path, required=True, help="directory containing source/mask PNGs")
    parser.add_argument("--output", type=Path, required=True, help="proposal JSON path")
    parser.add_argument("--min-coverage", type=float, default=.25)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.sides, args.patch_dir, args.output,
                         min_coverage=args.min_coverage), indent=2))


if __name__ == "__main__":
    main()
