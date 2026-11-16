"""Compile narrowly supported material-pattern and garage-leaf source updates.

This is an external candidate compiler. It never edits the active profile set.
Only explicit, source-hashed, high-confidence primary-surface pattern evidence
is eligible. Garage leaf state requires a projected opening to match a unique
open-state ROI; an open-looking dark bay or a prose note alone is insufficient.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import sys

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box_iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _overlap_fractions(a: list[float], b: list[float]) -> tuple[float, float, float]:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    ab = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return inter / max(aa, 1e-9), inter / max(ab, 1e-9), _box_iou(a, b)


def _opening_world_corners(profile: dict, opening: dict, ground_z: float) -> np.ndarray:
    normal = np.asarray(profile.get("normal"), dtype=float)
    if normal.shape != (2,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise ValueError("profile needs a finite facade normal")
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    frame = profile.get("frameU")
    if not isinstance(frame, list) or len(frame) != 2:
        raise ValueError("profile needs a two-value facade frame")
    height = float(profile.get("heightRefM", 0.))
    if height <= 0 or not math.isfinite(height):
        raise ValueError("profile needs a positive height reference")
    rect = opening.get("rect")
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        raise ValueError("opening needs a normalized [x0,z0,x1,z1] rectangle")
    x0, z0, x1, z1 = map(float, rect)
    style = opening.get("style") or {}
    recess = float(style.get("recessM", 0.) or 0.)
    plane = float(profile["planeV"]) - recess
    corners = []
    for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)):
        xy = normal * plane + tangent * (float(frame[0]) + x * (float(frame[1]) - float(frame[0])))
        corners.append([xy[0], xy[1], float(ground_z) + z * height])
    return np.asarray(corners, dtype=float)


def project_opening_to_perspective(profile: dict, opening: dict, *, ground_z: float,
                                   packet: dict, image_size: tuple[int, int],
                                   eye_height_m: float | None = None) -> list[float] | None:
    """Project a compiled opening using its exact packet camera and wall frame.

    The packet perspective crop uses the same wall-centred pinhole framing as
    ``regen/sides.py``. The camera XY/ground and facade frame come from the
    matched source packet/side. The returned rectangle is a projection and is
    not a physical measurement.
    """
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from frontage_crops_v15 import EYE_HEIGHT_M, view_basis, view_parameters

    camera_xy = np.asarray(packet.get("cameraXY"), dtype=float)
    camera_ground = float(packet.get("cameraGroundZ"))
    normal = np.asarray(profile.get("normal"), dtype=float)
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    frame = profile.get("frameU")
    midpoint = normal * float(profile["planeV"]) + tangent * (sum(map(float, frame)) / 2.)
    wall = {"midpoint": midpoint, "normal": normal,
            "lengthM": float(frame[1]) - float(frame[0]),
            "heightM": float(profile["heightRefM"])}
    if float((camera_xy - midpoint) @ normal) <= 0:
        return None
    vp = view_parameters(camera_xy, wall, float(ground_z))
    hfov = min(100., float(vp["hfovDeg"]) * 1.25)
    vfov = min(100., float(vp["vfovDeg"]) * 1.15)
    width, height = image_size
    fx = (width / 2.) / math.tan(math.radians(hfov) / 2.)
    expected_height = int(round(2. * fx * math.tan(math.radians(vfov) / 2.)))
    if expected_height > 1000:
        expected_height = 1000
        vfov = math.degrees(2. * math.atan((expected_height / 2.) / fx))
    if expected_height != height:
        # Require the recorded perspective dimensions to agree with the crop
        # projection rule; silently stretching boxes would create false matches.
        if abs(expected_height - height) > 2:
            return None
    right, down, forward = view_basis(vp["forward"])
    camera = np.r_[camera_xy, camera_ground + float(EYE_HEIGHT_M if eye_height_m is None else eye_height_m)]
    points = _opening_world_corners(profile, opening, ground_z)
    rays = points - camera
    depth = rays @ forward
    if np.any(depth <= 0):
        return None
    px = width / 2. + fx * (rays @ right) / depth
    py = height / 2. + fx * (rays @ down) / depth
    return [float(px.min()), float(py.min()), float(px.max()), float(py.max())]


def unique_roi_match(projected: list[list[float]], roi: list[float], *,
                     min_projected_coverage: float = .60,
                     min_roi_coverage: float = .60,
                     min_iou: float = .35,
                     runner_up_ratio: float = 1.5) -> dict[str, Any]:
    """Return one unambiguous opening/ROI match or an abstention reason."""
    scores = []
    for i, box in enumerate(projected):
        projected_cov, roi_cov, iou = _overlap_fractions(box, roi)
        scores.append({"index": i, "projectedCoverage": projected_cov,
                       "roiCoverage": roi_cov, "iou": iou,
                       "score": min(projected_cov, roi_cov) * iou})
    scores.sort(key=lambda s: s["score"], reverse=True)
    if not scores or scores[0]["projectedCoverage"] < min_projected_coverage or \
            scores[0]["roiCoverage"] < min_roi_coverage or scores[0]["iou"] < min_iou:
        return {"status": "abstain", "reason": "no substantial opening/ROI overlap", "scores": scores}
    if len(scores) > 1 and scores[0]["score"] < runner_up_ratio * max(scores[1]["score"], 1e-9):
        return {"status": "abstain", "reason": "ROI matches multiple openings", "scores": scores}
    return {"status": "matched", "openingIndex": scores[0]["index"], "scores": scores}


def apply_open_leaf(profile: dict, opening_index: int, observation: dict,
                    match: dict[str, Any]) -> bool:
    """Apply only explicit high-confidence open-state evidence with a unique projected match."""
    if match.get("status") != "matched" or observation.get("state") != "open" or \
            observation.get("confidence") != "high" or observation.get("ambiguity"):
        return False
    openings = profile.get("openings", [])
    if not (0 <= opening_index < len(openings)):
        return False
    style = openings[opening_index].get("style") or {}
    if style.get("kind") not in ("rollup", "garage"):
        return False
    style["doorKind"] = "open"
    style["openLeaf"] = True
    return True


def _apply_pattern(profile: dict, proposal: dict, packet: dict) -> tuple[bool, str]:
    evidence = profile.get("materialEvidence") or {}
    if proposal.get("field") != "materialEvidence.pattern":
        return False, "proposal does not target the existing primary material pattern"
    if proposal.get("confidence") != "high" or proposal.get("surfaceScope") != "primary_material_surface":
        return False, "pattern evidence is not high-confidence for a broad primary surface"
    if proposal.get("imageSha256") != packet.get("sourceSha256") or \
            str(proposal.get("imageId")) != str(packet.get("image")):
        return False, "pattern source image or hash does not match the exact packet camera"
    if evidence.get("family") not in ("timber", "brick", "shingle", "tile"):
        return False, "observed pattern is not compatible with the existing material family"
    if evidence.get("pattern"):
        return False, "existing profile already has a material pattern"
    pattern = proposal.get("value")
    allowed = {"timber": {"horizontal_courses", "vertical_boards", "scalloped_shingles"},
               "brick": {"running_bond", "stack_bond"},
               "shingle": {"horizontal_courses", "scalloped_shingles"},
               "tile": {"horizontal_courses", "vertical_courses"}}
    if pattern not in allowed[evidence["family"]]:
        return False, "pattern is not allowed for the existing family"
    evidence["pattern"] = pattern
    base_material = profile.get("material", "")
    color = "#" + "".join(f"{int(c):02x}" for c in evidence.get("colorRgb", [])) if len(evidence.get("colorRgb", [])) == 3 else None
    if color and base_material:
        from pipeline.material_recipes import make_observed_key
        profile["material"] = make_observed_key(base_material.split("#", 1)[0], color,
                                                family=evidence["family"], pattern=pattern,
                                                finish=evidence.get("finish"), moduleM=evidence.get("moduleM"))
    return True, ""


def compile_candidate(profiles: dict, material_audit: dict, access_inventory: dict,
                      packets_doc: dict, side_pieces: dict[str, dict], *,
                      eye_height_m: float | None = None) -> tuple[dict, dict]:
    """Return copied candidate profiles and a full audit for every source packet."""
    candidate = copy.deepcopy(profiles)
    by_profile = {p.get("id"): p for p in candidate.get("profiles", [])}
    material_by_side = {r.get("side"): r for r in material_audit.get("records", [])}
    access_by_side = {r.get("side"): r for r in access_inventory.get("records", [])}
    audited = []
    for packet in packets_doc.get("packets", []):
        side = packet.get("side")
        profile = by_profile.get("regen-" + str(side))
        mrecord, arecord = material_by_side.get(side), access_by_side.get(side)
        entry = {"side": side, "imageId": packet.get("image"),
                 "sourcePanoramaSha256": packet.get("sourceSha256"),
                 "profileFound": profile is not None, "updates": [], "abstentions": [],
                 "projectedGarageOpenings": []}
        if not profile or not mrecord or not arecord:
            entry["abstentions"].append("exact profile/material/access records are not all present")
            audited.append(entry)
            continue
        source_ids = {str(s).split(":", 1)[-1] for s in profile.get("sourceIds", [])}
        if str(packet.get("image")) not in source_ids:
            entry["abstentions"].append("packet image is not in the profile sourceIds")
            audited.append(entry)
            continue

        for proposal in mrecord.get("proposals", []):
            proposal_copy = copy.deepcopy(proposal)
            proposal_copy["imageId"] = mrecord.get("source", {}).get("imageId")
            proposal_copy["imageSha256"] = mrecord.get("source", {}).get("sourcePanoramaSha256")
            changed, reason = _apply_pattern(profile, proposal_copy, packet)
            if changed:
                entry["updates"].append({"kind": "material_pattern", "value": proposal_copy.get("value"),
                                         "imageId": packet.get("image"), "photoSha256": packet.get("sourceSha256"),
                                         "pixelROI": proposal_copy.get("pixelBounds"),
                                         "roiImage": proposal_copy.get("image"),
                                         "confidence": proposal_copy.get("confidence"),
                                         "ambiguity": proposal_copy.get("pixelBoundsMeaning")})
            elif proposal.get("field") == "materialEvidence.pattern":
                entry["abstentions"].append({"kind": "material_pattern", "reason": reason,
                                             "pixelROI": proposal.get("pixelBounds"),
                                             "confidence": proposal.get("confidence")})

        packet_view = Image.open(packet["originalPerspective"])
        image_size = packet_view.size
        side_piece = side_pieces.get(side)
        camera_view = next((v for v in (side_piece or {}).get("views", [])
                            if str(v.get("image")) == str(packet.get("image"))), None)
        if not side_piece or not camera_view:
            entry["abstentions"].append({"kind": "garage_leaf", "reason": "matched SID camera/ground record unavailable"})
            audited.append(entry)
            continue
        roi_features = [f for f in arecord.get("visualFeatures", []) if f.get("family") == "garage"]
        projections = []
        opening_refs = []
        for oi, opening in enumerate(profile.get("openings", [])):
            style = opening.get("style") or {}
            if style.get("kind") not in ("rollup", "garage"):
                continue
            try:
                box = project_opening_to_perspective(profile, opening,
                                                     ground_z=float(side_piece["groundZ"]),
                                                     packet=packet, image_size=image_size,
                                                     eye_height_m=eye_height_m)
            except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
                box = None
                entry["abstentions"].append({"kind": "garage_projection", "openingIndex": oi,
                                             "reason": str(exc)})
            if box is not None:
                projections.append(box)
                opening_refs.append(oi)
                entry["projectedGarageOpenings"].append({"openingIndex": oi, "kind": style.get("kind"),
                                                          "doorKind": style.get("doorKind"),
                                                          "projectedPixelBox": box,
                                                          "cameraImageId": packet.get("image"),
                                                          "cameraGroundZ": packet.get("cameraGroundZ"),
                                                          "facadeGroundZ": side_piece.get("groundZ"),
                                                          "recessM": style.get("recessM", 0.)})
        for observed in roi_features:
            roi = observed.get("pixelBounds")
            if not isinstance(roi, list) or len(roi) != 4:
                continue
            match = unique_roi_match(projections, roi)
            entry["garageROIComparisons"] = entry.get("garageROIComparisons", [])
            entry["garageROIComparisons"].append({"roi": roi, "roiImage": observed.get("image"),
                                                   "confidence": observed.get("confidence"),
                                                   "candidateMatch": observed.get("match"),
                                                   "sourceNote": observed.get("note"),
                                                   "doorLeafObservation": observed.get("doorLeafObservation"),
                                                   "projectionMatch": ({**match, "openingIndex": opening_refs[match["openingIndex"]]}
                                                                       if match.get("status") == "matched" else match)})
            # Inventory ROIs describe visible garage areas; they do not label
            # an observed leaf state. Require explicit state evidence as well.
            if match.get("status") != "matched":
                entry["abstentions"].append({"kind": "garage_leaf", "reason": match.get("reason"),
                                             "pixelROI": roi, "confidence": observed.get("confidence"),
                                             "doorLeafObservation": observed.get("doorLeafObservation")})
                continue
            index = opening_refs[match["openingIndex"]]
            state_observation = observed.get("doorState") or {}
            if not apply_open_leaf(profile, index, state_observation, match):
                entry["abstentions"].append({"kind": "garage_leaf", "openingIndex": index,
                                             "reason": "no high-confidence, unambiguous open-leaf observation is attached to the matched ROI",
                                             "pixelROI": roi, "confidence": state_observation.get("confidence"),
                                             "doorStateEvidence": state_observation or observed.get("doorLeafObservation")})
        for observed in roi_features:
            for state_observation in observed.get("doorLeafObservations", []):
                roi = state_observation.get("pixelBounds")
                if not isinstance(roi, list) or len(roi) != 4:
                    entry["abstentions"].append({"kind": "garage_leaf", "reason": "leaf ROI is missing valid pixel bounds"})
                    continue
                match = unique_roi_match(projections, roi)
                comparison = {"roi": roi, "roiImage": observed.get("image"),
                              "state": state_observation.get("state"),
                              "confidence": state_observation.get("confidence"),
                              "ambiguity": state_observation.get("ambiguity"),
                              "projectionMatch": ({**match, "openingIndex": opening_refs[match["openingIndex"]]}
                                                  if match.get("status") == "matched" else match)}
                entry.setdefault("doorLeafComparisons", []).append(comparison)
                if match.get("status") != "matched":
                    entry["abstentions"].append({"kind": "garage_leaf", "reason": match.get("reason"),
                                                 "pixelROI": roi, "confidence": state_observation.get("confidence"),
                                                 "doorStateEvidence": state_observation})
                    continue
                opening_index = opening_refs[match["openingIndex"]]
                if state_observation.get("state") == "open":
                    if not apply_open_leaf(profile, opening_index, state_observation, match):
                        entry["abstentions"].append({"kind": "garage_leaf", "openingIndex": opening_index,
                                                     "reason": "open-state evidence is not unambiguous and high-confidence",
                                                     "pixelROI": roi, "doorStateEvidence": state_observation})
                else:
                    entry["abstentions"].append({"kind": "garage_leaf", "openingIndex": opening_index,
                                                 "reason": "source ROI does not assert an open leaf; door state unchanged",
                                                 "pixelROI": roi, "doorStateEvidence": state_observation})
        if not roi_features and entry["projectedGarageOpenings"]:
            entry["abstentions"].append({"kind": "garage_leaf", "reason": "no source-annotated garage ROI or leaf state"})
        audited.append(entry)
    return candidate, {"schema": "source-semantics-compile-audit-v1", "reviewedCount": len(audited),
                       "updatedProfileCount": sum(bool(e["updates"]) for e in audited), "records": audited}


def _read_side_pieces(paths: list[Path]) -> dict[str, dict]:
    out = {}
    for path in paths:
        data = json.loads(path.read_text())
        for piece in data.get("pieces", []):
            out[piece.get("id")] = piece
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--material-audit", type=Path, required=True)
    ap.add_argument("--access-inventory", type=Path, required=True)
    ap.add_argument("--packets", type=Path, required=True)
    ap.add_argument("--sides", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    inputs = [args.profiles, args.material_audit, args.access_inventory, args.packets, *args.sides]
    if args.output.exists():
        ap.error("output directory already exists; choose a new candidate directory")
    profiles, materials = json.loads(args.profiles.read_text()), json.loads(args.material_audit.read_text())
    access, packets = json.loads(args.access_inventory.read_text()), json.loads(args.packets.read_text())
    candidate, audit = compile_candidate(profiles, materials, access, packets, _read_side_pieces(args.sides))
    args.output.mkdir(parents=True)
    candidate["sourceSemanticsCandidate"] = {"basedOn": str(args.profiles), "updates": audit["updatedProfileCount"]}
    (args.output / "profiles-v18-source-semantics-candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    audit["inputs"] = [{"path": str(path), "sha256": sha256(path)} for path in inputs]
    (args.output / "source-semantics-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"reviewedCount": audit["reviewedCount"], "updatedProfileCount": audit["updatedProfileCount"],
                      "candidate": str(args.output / "profiles-v18-source-semantics-candidate.json"),
                      "audit": str(args.output / "source-semantics-audit.json")}, indent=2))


if __name__ == "__main__":
    main()
