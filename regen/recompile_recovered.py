#!/usr/bin/env python3
"""Recompile source-qualified recovered readings without replacing unresolved profiles.

Each candidate must match the existing profile's source image, notes and confidence.
Candidates are grouped by a semantic hash that excludes only prose/audit fields. A side
is staged only when one semantic reading remains and the source view/frame/crop files
pass reading_provenance validation. The profile compiler then applies its normal
registration and schema transformations. A final source directory may be applied last
to preserve reviewed overlays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reading_provenance as provenance


SEMANTIC_OMIT = {
    "notes", "confidence", "audit", "provenance", "inventory", "unmodelled",
    "reviewedCategories", "unresolved", "readingCoverage", "warnings",
    "commentary", "rationale", "sourceNotes",
}
GEOMETRY_FIELDS = {
    "panels", "groups", "openings", "bays", "bands", "piers", "parapets",
    "recesses", "signs", "canopies", "ornaments", "balconies", "roofs",
    "streetEdge", "planting", "rustication", "quoins", "fireEscapes", "doors",
    "material", "materialEvidence",
    "normal", "frameU", "planeV", "heightRefM", "zBaseM",
}
READING_DOMAIN_FIELDS = {"wallTopM", "storeys"}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_reading(reading):
    return {k: v for k, v in reading.items() if k not in SEMANTIC_OMIT}


def semantic_hash(reading):
    return canonical_hash(semantic_reading(reading))


def geometry_hash(profile):
    reading = profile.get("reading") or {}
    profile_geometry = {k: profile.get(k) for k in sorted(GEOMETRY_FIELDS)}
    # The current compiler adds an empty optional signs array; absence and [] mean
    # the same geometry when comparing against the v14 base.
    if profile_geometry.get("signs") == []:
        profile_geometry["signs"] = None
    return canonical_hash({
        "profile": profile_geometry,
        "readingDomain": {k: reading.get(k) for k in sorted(READING_DOMAIN_FIELDS)},
    })


def relative(path, root):
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_match(candidate, profile, nominal, registered, calibrated, area, camera_calibration=None):
    """Validate source image, registered frame, and required crop assets."""
    sid = profile["id"].removeprefix("regen-")
    reading = candidate["reading"]
    use = reading.get("use", 0)
    if isinstance(use, bool) or not isinstance(use, int):
        return None, "reading use is not an integer"
    calibrated_crop = bool(reading.get("calibratedCrops"))
    registered_required = bool(reading.get("registeredCrops"))
    side_index = calibrated if calibrated_crop else registered if registered_required else {}
    selection = provenance.select_source_view(
        nominal[sid], side_index, use, registered_required=registered_required,
    )
    if selection.get("status") != "ok":
        return None, selection.get("reason", "source view unresolved")
    expected = set(profile.get("sourceIds") or [])
    actual = set(selection.get("sourceIds") or [])
    if len(expected) != 1 or actual != expected:
        return None, f"source image mismatch: profile={sorted(expected)} selected={sorted(actual)}"
    if calibrated_crop and not (selection.get("view") or {}).get("gravityCalibration"):
        return None, "calibrated reading has no matching gravityCalibration crop metadata"
    if camera_calibration and not calibrated_crop:
        record = camera_calibration.get("images", {}).get((selection.get("view") or {}).get("image"))
        if record and record.get("applied"):
            view = selection.get("view") or {}
            if view.get("registered", False) != record.get("registration", False):
                return None, "camera calibration and crop placement modes differ"
            if view.get("cameraGroundZ") is None:
                return None, "camera calibration crop has no cameraGroundZ"
    side_path = selection.get("side_path")
    crop_root = (Path(side_path).parent / "img") if side_path else (area / "regen/sides/img")
    missing = provenance.missing_crop_files(selection, crop_root)
    if missing:
        return None, "missing crop assets: " + ", ".join(missing)
    return selection, None


def run_profile_compiler(python, script, area, base, readings, out, registered, calibrated,
                         components=None, camera_calibration=None):
    command = [str(python), str(script), str(area), str(base), str(readings), str(out), "--keep-low",
               "--registered-sides", str(registered)]
    if calibrated:
        command += ["--calibrated-sides", str(calibrated)]
    if components:
        command += ["--component-observations", str(components)]
    if camera_calibration:
        command += ["--camera-calibration", str(camera_calibration)]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(
            f"profile compiler failed ({proc.returncode}): {' '.join(command)}\n{proc.stderr[-4000:]}"
        )
    return proc.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path, help="Existing profile JSON to preserve on unresolved sides")
    ap.add_argument("--candidate-dir", action="append", required=True, type=Path,
                    help="Directory of legacy readings; repeat for every archive/overlay to search")
    ap.add_argument("--stage-dir", required=True, type=Path, help="Persistent directory for selected source readings")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--profile-script", required=True, type=Path)
    ap.add_argument("--registered-sides", required=True, type=Path)
    ap.add_argument("--calibrated-sides", type=Path)
    ap.add_argument("--fresh-overlay", type=Path,
                    help="Optional reviewed reading directory applied after recovered legacy readings")
    ap.add_argument("--fresh-components", type=Path)
    ap.add_argument("--camera-calibration", type=Path,
                    help="Optional correction manifest applied once to uncalibrated readings in both passes")
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--overwrite", action="store_true", help="Replace output/stage artifacts from a prior run")
    args = ap.parse_args()

    area, base = args.area.resolve(), args.base.resolve()
    profile_script = args.profile_script.resolve()
    registered_dir = args.registered_sides.resolve()
    calibrated_dir = args.calibrated_sides.resolve() if args.calibrated_sides else None
    camera_calibration = args.camera_calibration.resolve() if args.camera_calibration else None
    stage_dir = args.stage_dir.resolve()
    manifest_path, output_path = args.manifest.resolve(), args.out.resolve()
    for label, path in (("area", area), ("base", base), ("profile script", profile_script),
                        ("registered sides", registered_dir)):
        if not path.exists():
            ap.error(f"{label} does not exist: {path}")
    if stage_dir.exists() and any(stage_dir.iterdir()) and not args.overwrite:
        ap.error(f"stage directory is not empty (use --overwrite to replace): {stage_dir}")
    if args.overwrite and stage_dir.exists():
        for child in stage_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil
                shutil.rmtree(child)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_doc = read_json(base)
    base_profiles = base_doc.get("profiles", [])
    profiles_by_side = {p["id"].removeprefix("regen-"): p for p in base_profiles}
    nominal_doc = read_json(area / "regen/sides/sides.json")
    nominal = {p["id"]: p for p in nominal_doc.get("pieces", [])}
    registered = provenance.load_registered_sides(registered_dir)
    calibrated = provenance.load_registered_sides(calibrated_dir) if calibrated_dir else {}
    camera_calibration_data = read_json(camera_calibration) if camera_calibration else None

    candidate_map = {}
    candidate_dirs = [p.resolve() for p in args.candidate_dir]
    for directory in candidate_dirs:
        if not directory.is_dir():
            ap.error(f"candidate directory does not exist: {directory}")
        for path in sorted(directory.glob("*.json")):
            try:
                reading = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            sid = reading.get("id") if isinstance(reading, dict) else None
            if isinstance(sid, str):
                candidate_map.setdefault(sid, []).append({"path": path, "reading": reading})

    decisions = []
    staged = []
    semantic_counts = Counter()
    for sid, profile in profiles_by_side.items():
        # Do not reselect profiles which already carry explicit reading provenance.
        if profile.get("readingProvenance"):
            continue
        if sid not in nominal:
            decisions.append({"side": sid, "status": "unresolved_existing_profile_preserved",
                              "reason": "nominal side frame missing"})
            semantic_counts["unresolved"] += 1
            continue
        expected_reading = profile.get("reading") or {}
        guarded = [c for c in candidate_map.get(sid, [])
                   if c["reading"].get("notes") == expected_reading.get("notes")
                   and c["reading"].get("confidence") == expected_reading.get("confidence")]
        if not guarded:
            decisions.append({"side": sid, "status": "unresolved_existing_profile_preserved",
                              "reason": "no candidate matches profile source notes and confidence"})
            semantic_counts["no_notes_match"] += 1
            continue
        valid = []
        invalid = []
        for candidate in guarded:
            selection, error = source_match(candidate, profile, nominal, registered, calibrated, area,
                                            camera_calibration_data)
            if error:
                invalid.append({"path": relative(candidate["path"], area), "reason": error})
                continue
            candidate["selection"] = selection
            candidate["semanticHash"] = semantic_hash(candidate["reading"])
            valid.append(candidate)
        by_semantic = {}
        for candidate in valid:
            by_semantic.setdefault(candidate["semanticHash"], []).append(candidate)
        if not by_semantic:
            decisions.append({"side": sid, "status": "unresolved_existing_profile_preserved",
                              "reason": "candidate source/frame/crop validation failed",
                              "candidates": invalid})
            semantic_counts["source_validation_failed"] += 1
            continue
        if len(by_semantic) > 1:
            decisions.append({"side": sid, "status": "ambiguous_existing_profile_preserved",
                              "reason": "multiple source-valid geometry-bearing semantic readings remain",
                              "semanticVariants": [{"semanticSha256": h,
                                                    "paths": [relative(c["path"], area) for c in cs]}
                                                   for h, cs in sorted(by_semantic.items())],
                              "invalidCandidates": invalid})
            semantic_counts["ambiguous"] += 1
            continue
        semantic, copies = next(iter(by_semantic.items()))
        # Same-semantic files are equivalent on typed geometry, crop flags, use and source metadata.
        # A stable lexical pick avoids treating a newer directory as inherently authoritative.
        chosen = sorted(copies, key=lambda c: str(c["path"]))[0]
        staged_path = stage_dir / f"{sid}.json"
        staged_path.write_text(json.dumps(chosen["reading"], indent=2, ensure_ascii=False) + "\n")
        staged.append(sid)
        semantic_counts["staged"] += 1
        decisions.append({
            "side": sid,
            "status": "staged_source_valid_semantic_unique",
            "semanticSha256": semantic,
            "chosenPath": relative(chosen["path"], area),
            "chosenAbsolutePath": str(chosen["path"].resolve()),
            "chosenSha256": provenance.sha256_file(chosen["path"]),
            "duplicateEquivalentPaths": [relative(c["path"], area) for c in copies if c is not chosen],
            "sourceImage": (chosen["selection"].get("sourceIds") or [None])[0],
            "sourceSide": relative(chosen["selection"]["side_path"], area)
            if chosen["selection"].get("side_path") else "regen/sides/sides.json",
            "registeredCropSource": bool(chosen["reading"].get("registeredCrops")),
            "calibratedCropSource": bool(chosen["reading"].get("calibratedCrops")),
            "frameValidation": "matched",
            "cropFiles": {k: chosen["selection"]["view"].get(k) for k in ("rect", "persp")},
            "invalidOtherCandidates": invalid,
        })

    fresh = args.fresh_overlay.resolve() if args.fresh_overlay else None
    if fresh:
        for path in sorted(fresh.glob("*.json")):
            try:
                reading = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            sid = reading.get("id") if isinstance(reading, dict) else None
            if not isinstance(sid, str) or sid not in profiles_by_side:
                continue
            selection, error = source_match(
                {"path": path, "reading": reading}, profiles_by_side[sid], nominal,
                registered, calibrated, area, camera_calibration_data,
            )
            if error:
                raise RuntimeError(f"fresh overlay source/calibration validation failed for {sid}: {error}")

    scratch = output_path.parent / (output_path.stem + ".recovered-intermediate.json")
    stage_stdout = run_profile_compiler(args.python, profile_script, area, base, stage_dir, scratch,
                                        registered_dir, calibrated_dir,
                                        camera_calibration=camera_calibration)
    output_stdout = stage_stdout
    fresh_count = 0
    if fresh:
        if not fresh.is_dir():
            ap.error(f"fresh overlay does not exist: {fresh}")
        fresh_count = 0
        for path in fresh.glob("*.json"):
            try:
                fresh_count += isinstance(read_json(path), dict) and isinstance(read_json(path).get("id"), str)
            except (OSError, json.JSONDecodeError):
                continue
        output_stdout = run_profile_compiler(args.python, profile_script, area, scratch, fresh,
                                             output_path, registered_dir, calibrated_dir,
                                             args.fresh_components, camera_calibration)
    else:
        scratch.replace(output_path)

    result_doc = read_json(output_path)
    result_by_side = {p["id"].removeprefix("regen-"): p for p in result_doc.get("profiles", [])}
    base_by_side = profiles_by_side
    changed = []
    geometry_changed = material_changed = unchanged_preserved = dropped = 0
    for sid, old in base_by_side.items():
        new = result_by_side.get(sid)
        if new is None:
            dropped += 1
            continue
        if canonical_hash(old) == canonical_hash(new):
            unchanged_preserved += 1
            continue
        changed.append(sid)
        if any(canonical_hash(old.get(k)) != canonical_hash(new.get(k)) for k in GEOMETRY_FIELDS - {"material", "materialEvidence"}):
            geometry_changed += 1
        if any(canonical_hash(old.get(k)) != canonical_hash(new.get(k)) for k in ("material", "materialEvidence", "panels")):
            material_changed += 1

    fresh_ids = []
    if args.fresh_overlay:
        for path in Path(args.fresh_overlay).glob("*.json"):
            try:
                record = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                fresh_ids.append(record["id"])
    fresh_geometry_diffs = []
    for sid in sorted(set(fresh_ids)):
        old, new = base_by_side.get(sid), result_by_side.get(sid)
        if old is None or new is None:
            fresh_geometry_diffs.append({"side": sid, "reason": "fresh profile missing in base or result"})
        elif geometry_hash(old) != geometry_hash(new):
            fresh_geometry_diffs.append({
                "side": sid,
                "baseGeometrySha256": geometry_hash(old),
                "resultGeometrySha256": geometry_hash(new),
                "baseFrame": {k: old.get(k) for k in ("normal", "frameU", "planeV", "heightRefM", "zBaseM")},
                "resultFrame": {k: new.get(k) for k in ("normal", "frameU", "planeV", "heightRefM", "zBaseM")},
                "baseWallDomain": {k: (old.get("reading") or {}).get(k) for k in sorted(READING_DOMAIN_FIELDS)},
                "resultWallDomain": {k: (new.get("reading") or {}).get(k) for k in sorted(READING_DOMAIN_FIELDS)},
            })

    summary = {
        "profileCountBefore": len(base_profiles),
        "profileCountAfter": len(result_doc.get("profiles", [])),
        "candidateFiles": sum(len(v) for v in candidate_map.values()),
        "stagedSemanticUnique": len(staged),
        "ambiguousOrUnresolvedPreserved": semantic_counts["ambiguous"] + semantic_counts["source_validation_failed"] + semantic_counts["no_notes_match"] + semantic_counts["unresolved"],
        "freshOverlayFiles": fresh_count,
        "profilesChanged": len(changed),
        "geometryChanged": geometry_changed,
        "materialChanged": material_changed,
        "unchangedProfilesPreserved": unchanged_preserved,
        "profilesDroppedByCompiler": dropped,
        "freshOverlayProfiles": len(set(fresh_ids)),
        "freshOverlayGeometryChangedVsBase": len(fresh_geometry_diffs),
    }
    manifest = {
        "schema": "recovered-profile-recompile-v1",
        "result": str(output_path),
        "base": {"path": str(base), "sha256": provenance.sha256_file(base)},
        "candidateDirectories": [str(p) for p in candidate_dirs],
        "stageDirectory": str(stage_dir),
        "freshOverlay": str(args.fresh_overlay.resolve()) if args.fresh_overlay else None,
        "componentObservation": {"path": str(args.fresh_components.resolve()),
                                 "sha256": provenance.sha256_file(args.fresh_components)}
        if args.fresh_components else None,
        "registeredSides": str(registered_dir),
        "calibratedSides": str(calibrated_dir) if calibrated_dir else None,
        "cameraCalibration": {"path": str(camera_calibration), "sha256": provenance.sha256_file(camera_calibration)}
        if camera_calibration else None,
        "semanticProjectionExcludedFields": sorted(SEMANTIC_OMIT),
        "counts": {**summary, "decisionCounts": dict(semantic_counts)},
        "compilerStdout": output_stdout,
        "profilesChanged": changed,
        "freshOverlayGeometryDiffs": fresh_geometry_diffs,
        "decisions": decisions,
        "unresolvedOrAmbiguous": [d for d in decisions if d["status"] != "staged_source_valid_semantic_unique"],
        "claims": {"sourceAndCropMappingsValidated": True,
                   "recoveredDoesNotMeanImageVerified": True,
                   "ambiguousOrInvalidSidesKeepTheirBaseGeometry": True,
                   "freshOverlayAppliedLast": bool(args.fresh_overlay)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if scratch.exists():
        scratch.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
