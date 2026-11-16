"""Apply local semantic/opening masks to high-signal module candidates only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from facade_segment_v27 import Detector, Segmenter
from regen.appearance_patches import detector_valid
from regen.measure_surface_modules import measure_view, compare_views


def strong_axis(view: dict, key: str, *, min_period_px: float = 8.0) -> dict:
    """Require three agreeing strips and an adequately sampled physical repeat."""
    axis = view.get("axes", {}).get(key, {})
    strips = [s for s in axis.get("strips", []) if s.get("accepted") and float(s.get("period_px", 0)) >= min_period_px]
    if len(strips) < 3:
        return {"accepted": False, "acceptedStrips": len(strips), "minimumPeriodPx": min_period_px,
                "reason": "fewer than three separated strips meet the 8 px anti-alias margin"}
    periods = np.asarray([float(s["period_px"]) for s in strips])
    median = float(np.median(periods))
    mad = float(np.median(np.abs(periods - median)) / max(median, 1e-9))
    physical = median / float(view["ppm"])
    accepted = mad <= .12 and .06 <= physical <= .40
    return {"accepted": accepted, "period_px": round(median, 3), "period_m": round(physical, 5),
            "acceptedStrips": len(strips), "stripAgreementMadFraction": round(mad, 4),
            "minimumPeriodPx": min_period_px,
            "reason": None if accepted else "strip dispersion or timber-board pitch range failed"}


def run(candidate_report: Path, output_dir: Path, output_path: Path) -> dict:
    report = json.loads(candidate_report.read_text())
    candidates = [r for r in report["records"] if r.get("accepted")]
    segmenter, detector = Segmenter(), Detector()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    for rec in candidates:
        semantic_views = []
        file_metas = [p for p in rec.get("patches", []) if p.get("diagnosticFile")]
        for vi, meta in enumerate(file_metas[:2]):
            rgba = np.asarray(Image.open(meta["diagnosticFile"]).convert("RGBA"))
            rgb = rgba[..., :3]
            valid = rgba[..., 3] > 0
            semantic = segmenter.labels(rgb)
            valid &= semantic == 0
            boxes = detector.boxes(rgb)
            valid &= detector_valid(valid.shape, boxes)
            ppm = float(rec["viewResults"][vi]["ppm"])
            measurement = measure_view(rgb, valid, ppm)
            vertical = strong_axis(measurement, "vertical_repeat_px")
            horizontal = strong_axis(measurement, "horizontal_repeat_px")
            measurement["semanticCoverage"] = round(float(valid.mean()), 5)
            measurement["semanticLabels"] = {"retainedSurface": 0, "opening": 1, "sign": 2, "occluder": 3, "sky": 4}
            measurement["detectedOpeningBoxes"] = [{"xyxy": [round(float(x), 1) for x in b[:4]], "kind": b[4],
                                                     "score": round(float(b[5]), 4)} for b in boxes]
            measurement["verifiedVertical"] = vertical
            measurement["verifiedHorizontalControl"] = horizontal
            semantic_views.append(measurement)
            stem = f"{rec['featureId']}__{rec['profileId'].replace('/', '_')}__view{vi}"
            masked = Image.fromarray(np.dstack([rgb, np.where(valid, 255, 0).astype(np.uint8)]), "RGBA")
            target = output_dir / f"{stem}.png"
            masked.save(target)
            preview = Image.new("RGBA", rgb.shape[1::-1], (176, 176, 176, 255))
            preview.alpha_composite(masked)
            tile = preview.convert("RGB")
            tile.thumbnail((420, 260))
            card = Image.new("RGB", (440, 300), "white")
            card.paste(tile, ((440 - tile.width) // 2, 24))
            ImageDraw.Draw(card).text((8, 5), f"{rec['featureId']} img={meta.get('sourceImageId')}", fill="black")
            cards.append(card)
        if len(semantic_views) != 2:
            records.append({"featureId": rec["featureId"], "profileId": rec["profileId"], "accepted": False,
                            "reason": "two candidate patch files unavailable", "semanticViews": semantic_views})
            continue
        a = {"ppm": semantic_views[0]["ppm"], "axes": {"vertical_repeat_px": semantic_views[0]["verifiedVertical"],
                                                         "horizontal_repeat_px": semantic_views[0]["verifiedHorizontalControl"]}}
        b = {"ppm": semantic_views[1]["ppm"], "axes": {"vertical_repeat_px": semantic_views[1]["verifiedVertical"],
                                                         "horizontal_repeat_px": semantic_views[1]["verifiedHorizontalControl"]}}
        agreement = compare_views(a, b)
        # Do not infer boards if a stable horizontal-axis pattern suggests vertical boards/details.
        horizontal_alternative = agreement["axes"]["horizontal_repeat_px"]["accepted"]
        vertical = agreement["axes"]["vertical_repeat_px"]
        accepted = bool(vertical["accepted"] and not horizontal_alternative and
                        all(float(v.get("semanticCoverage", 0)) >= .25 for v in semantic_views))
        records.append({"featureId": rec["featureId"], "profileId": rec["profileId"],
                        "claimedPattern": rec.get("claimedPattern"),
                        "proposedPattern": "inferred_horizontal_periodic_boards" if accepted and not rec.get("claimedPattern") else None,
                        "accepted": accepted, "coursePitchM": vertical.get("period_m") if accepted else None,
                        "semanticViews": semantic_views, "agreement": agreement,
                        "reason": None if accepted else "paired geometry/semantic masks, 8 px alias margin, two-patch consensus, board-pitch range, or orientation alternative failed"})
    if cards:
        cols = 4
        sheet = Image.new("RGB", (cols * 440, ((len(cards) + cols - 1) // cols) * 300), "#dddddd")
        for i, card in enumerate(cards):
            sheet.paste(card, ((i % cols) * 440, (i // cols) * 300))
        sheet.save(output_dir / "contact-sheet.jpg", quality=90)
    result = {"schema": "semantic-horizontal-module-review-v1", "candidateSource": str(candidate_report),
              "candidateCount": len(candidates), "acceptedCount": sum(bool(r.get("accepted")) for r in records),
              "records": records,
              "gates": {"twoIndependentPhotos": True, "separatedStripsPerPhoto": "at least 3 of 4", "minRepeatPixels": 8,
                        "physicalBoardPitchRangeM": [.06, .40], "verticalAlternativeMustFail": True,
                        "semantic": "SegFormer retained-wall pixels plus local Grounding DINO opening/sign/awning boxes"},
              "limitations": ["semantic labels and detections still require visual mask review",
                              "no profile or material evidence was updated"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(a.candidates, a.output_dir, a.output)
    print(json.dumps({"candidates": r["candidateCount"], "accepted": r["acceptedCount"],
                      "features": [{"featureId": x["featureId"], "accepted": x["accepted"], "coursePitchM": x.get("coursePitchM")} for x in r["records"]]}, indent=2))


if __name__ == "__main__":
    main()
