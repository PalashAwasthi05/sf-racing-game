"""Promote measured cluster alignments to graded world registration.

The missing step. Every producer in this workspace writes
``worldRegistrationValidated: False`` as a hardcoded literal
(mapillary_normalize.py:42, mapillary_world_alignment.py:419,
rectify_facade_diagnostics_v12.py:695/728, propose_facade_parts_v12.py:341/395,
mask_facade_obstructions_v12.py:172, mapillary_plane_refinement.py:198), and
general_world/evidence.py rejects any observation whose camera does not carry
True. Nothing in the package ever writes True except a unit-test fixture
(test_detail_pipeline_v11.py:12). So the admission gate is not failing on
quality: it was never wired to open. That is why 8,631 discovered images and 160
downloaded panoramas produced zero image-derived facades.

The alignment diagnostic already holds the evidence. Of 41 clusters, 23 reach
``status: plausible, accepted: True`` with held-out medians near 1.2 m and 5 m
inlier ratios of 0.76-0.98. Coarse registration works.

WHAT THIS DOES NOT DO. It does not declare those clusters facade-grade, because
the same report says they are not: the best held-out inlier ratio at 0.5 m is
0.37 and at 0.25 m is 0.17. You cannot place a window reveal with that. So this
step does not flip a boolean; it records a GRADE measured from the held-out
split, and only the grade the data earns:

    coarse   held-out p75 <= COARSE_P75_M and held-out inlier@1m >= COARSE_AT1M
             enough to say WHICH surface an observation belongs to and roughly
             where on it: storey bands, bay divisions, material regions.

    fine     reserved, defined, and DELIBERATELY UNREACHABLE on this evidence.
             Nothing in the current data meets it. The grade exists so the
             opening path stays closed by measurement rather than by a literal,
             and so a later, better registration can open it without editing
             this file's intent.

The thresholds are read off the measured distribution rather than chosen to pass
a batch: the accepted clusters separate cleanly into four with held-out p75
under 1.35 m and inlier@1m above 0.62, and the rest at p75 2.0-4.1 m with
inlier@1m 0.27-0.52. COARSE_P75_M and COARSE_AT1M sit in that gap. The gap and
the counts either side are written into the report so the choice can be argued
with.

No location-specific logic: clusters are selected by measured alignment quality
alone, and no identity, street or building name takes part.

Writes a NEW index beside the original and changes nothing in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Grade thresholds, read off the measured held-out distribution (see module doc).
COARSE_P75_M = 1.5
COARSE_AT1M = 0.60
# Facade openings need sub-half-metre agreement before a contour means anything.
FINE_P75_M = 0.35
FINE_AT05M = 0.80
# A single-image cluster has no internal geometry to be wrong about; require a
# baseline before trusting its alignment.
MIN_IMAGES = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdout(reg: dict) -> dict:
    initial = reg.get("initial") or {}
    return initial.get("holdout") or initial.get("train") or {}


def grade_for(reg: dict, image_count: int) -> tuple[str, dict]:
    """Grade one cluster from its HELD-OUT split only."""
    stats = _holdout(reg)
    p75 = stats.get("p75M")
    ratios = stats.get("inlierRatioAtM") or {}
    at1 = ratios.get("1.0")
    at05 = ratios.get("0.5")
    measured = {
        "holdoutP75M": p75,
        "holdoutInlierRatio1m": at1,
        "holdoutInlierRatio0_5m": at05,
        "imageCount": image_count,
    }
    if not reg.get("accepted"):
        return "unregistered", {**measured, "reason": "alignment not accepted"}
    if image_count < MIN_IMAGES:
        return "unregistered", {**measured, "reason": f"fewer than {MIN_IMAGES} images"}
    if p75 is None or at1 is None:
        return "unregistered", {**measured, "reason": "no held-out statistics"}
    if p75 <= FINE_P75_M and (at05 or 0.0) >= FINE_AT05M:
        return "fine", measured
    if p75 <= COARSE_P75_M and at1 >= COARSE_AT1M:
        return "coarse", measured
    return "unregistered", {
        **measured,
        "reason": f"held-out p75 {p75:.2f} m / inlier@1m {at1:.3f} below the coarse grade",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", type=Path, default=Path("work/mapillary-street-alignment-v12.json"))
    ap.add_argument("--source-index", type=Path, default=Path("work/imagery-street-v12/source-index.json"))
    ap.add_argument("--out", type=Path, default=Path("work/imagery-street-v12/source-index-registered.json"))
    ap.add_argument("--report", type=Path, default=Path("outputs/world-registration-promotion-v14.json"))
    args = ap.parse_args()

    alignment = json.loads(args.alignment.read_text())
    index = json.loads(args.source_index.read_text())

    per_cluster_images: dict[str, int] = {}
    for rec in index.values():
        cid = rec.get("camera", {}).get("clusterId")
        if cid:
            per_cluster_images[cid] = per_cluster_images.get(cid, 0) + 1

    clusters = {}
    for entry in alignment.get("clusters", []):
        cid = entry.get("clusterId")
        reg = entry.get("registration")
        if not cid or not reg:
            clusters[cid] = {"grade": "unregistered", "measured": {"reason": "cluster could not be assessed"}}
            continue
        grade, measured = grade_for(reg, per_cluster_images.get(cid, 0))
        clusters[cid] = {"grade": grade, "measured": measured}

    promoted = {"fine": 0, "coarse": 0, "unregistered": 0}
    out_index = {}
    for key, rec in index.items():
        rec = json.loads(json.dumps(rec))  # copy; never mutate the input in place
        cam = rec.setdefault("camera", {})
        cid = cam.get("clusterId")
        info = clusters.get(cid, {"grade": "unregistered", "measured": {"reason": "no cluster"}})
        grade = info["grade"]
        # The provider's own pose consistency is an independent check and stays
        # a precondition: a cluster can align well while one shot inside it does
        # not agree with its own provider pose.
        if grade != "unregistered" and not cam.get("providerPoseConsistent"):
            grade = "unregistered"
            info = {**info, "measured": {**info["measured"], "reason": "provider pose not consistent"}}
        cam["registrationGrade"] = grade
        cam["registrationEvidence"] = info["measured"]
        # The boolean the existing gate reads. True only where a grade was
        # earned; the grade itself is what callers should branch on.
        cam["worldRegistrationValidated"] = grade in ("coarse", "fine")
        rec["admission"] = (
            f"world-registration-{grade}" if grade != "unregistered" else "pose-registration-required"
        )
        promoted[grade] += 1
        out_index[key] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_index, indent=2))

    graded_clusters = {
        g: sorted(cid for cid, v in clusters.items() if v["grade"] == g and cid)
        for g in ("fine", "coarse", "unregistered")
    }
    report = {
        "schema": "world-registration-promotion/v1",
        "purpose": (
            "the missing promotion step: every producer hardcoded "
            "worldRegistrationValidated False and nothing ever wrote True, so the "
            "evidence gate could not open regardless of alignment quality"
        ),
        "thresholds": {
            "COARSE_P75_M": COARSE_P75_M,
            "COARSE_AT1M": COARSE_AT1M,
            "FINE_P75_M": FINE_P75_M,
            "FINE_AT05M": FINE_AT05M,
            "MIN_IMAGES": MIN_IMAGES,
            "derivation": (
                "accepted clusters separate into four with held-out p75 <= 1.35 m and "
                "inlier@1m >= 0.62, and the remainder at p75 2.0-4.1 m with inlier@1m "
                "0.27-0.52; the coarse thresholds sit in that gap"
            ),
        },
        "inputs": {
            "alignment": str(args.alignment),
            "alignmentSha256": _sha256(args.alignment),
            "sourceIndex": str(args.source_index),
            "sourceIndexSha256": _sha256(args.source_index),
        },
        "images": {"total": len(index), **promoted},
        "clusters": {g: len(v) for g, v in graded_clusters.items()},
        "clusterIdsByGrade": graded_clusters,
        "perCluster": clusters,
        "fineGradeAchieved": promoted["fine"] > 0,
        "openingContoursAdmissible": promoted["fine"] > 0,
        "limitations": [
            "Coarse grade supports surface and region attribution, not opening contours: "
            "the best held-out inlier ratio at 0.5 m across all accepted clusters is 0.37.",
            "No fine-grade cluster exists on this evidence, so window, door and void "
            "records remain inadmissible by measurement rather than by a literal.",
            "Held-out statistics come from the existing alignment diagnostic and inherit "
            "its assumptions, including its vertical datum handling.",
            "This step writes a new index beside the original; nothing was changed in place, "
            "and no world, renderer or facade state was touched.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "images": report["images"],
        "clusters": report["clusters"],
        "fineGradeAchieved": report["fineGradeAchieved"],
        "wroteIndex": str(args.out),
        "wroteReport": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
