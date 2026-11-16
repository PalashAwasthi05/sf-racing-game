"""Audit horizontal timber course pitch from paired registered facade crops.

Local, diagnostic-only experiment. Uses readings' measured wall zones and
opening/projection masks, then requires repeated signal in spatially separated
strips and agreement between two registered photographs. It never edits profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from regen.appearance_patches import crop_zone, exclusion_mask
from regen.measure_surface_modules import measure_view, compare_views


def run(profiles_path: Path, readings_dir: Path, sides_paths: list[Path], image_dir: Path,
        out_dir: Path, output_path: Path) -> dict:
    profiles = json.loads(profiles_path.read_text())["profiles"]
    sides = {}
    for path in sides_paths:
        doc = json.loads(path.read_text())
        sides.update({p["id"]: p for p in doc.get("pieces", [])})
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    thumbs = []
    for profile in profiles:
        evidence = profile.get("materialEvidence", {})
        material_base = str(profile.get("material", "")).split("#", 1)[0].split("~", 1)[0]
        is_timber = evidence.get("family") == "timber" or material_base.startswith(("timber_dark", "timber_tan"))
        if not is_timber:
            continue
        pid = profile["id"].removeprefix("regen-")
        side = sides.get(pid)
        provenance = profile.get("readingProvenance", {}).get("sourceReading", {})
        src = provenance.get("path", "")
        if not side or len(side.get("views", [])) < 2 or not src:
            records.append({"featureId": profile.get("match", {}).get("featureId"), "profileId": profile["id"],
                            "accepted": False, "reason": "paired registered crops or source reading unavailable"})
            continue
        reading_path = readings_dir / Path(src).name
        if not reading_path.is_file():
            archive = next((part for part in Path(src).parts if part.startswith("readings-")), None)
            if archive:
                candidate = readings_dir.parent / archive / Path(src).name
                if candidate.is_file():
                    reading_path = candidate
        if not reading_path.is_file():
            # Source archives are all local; resolve by recorded archive + filename.
            matches = list(readings_dir.parent.rglob(Path(src).name))
            if matches:
                expected = provenance.get("sha256")
                reading_path = next((p for p in matches if expected and hashlib.sha256(p.read_bytes()).hexdigest() == expected), matches[0])
        if not reading_path.is_file():
            records.append({"featureId": profile.get("match", {}).get("featureId"), "profileId": profile["id"],
                            "accepted": False, "reason": "source reading file unavailable"})
            continue
        reading = json.loads(reading_path.read_text())
        zones = [z for z in reading.get("materials", []) if z.get("family") == "timber"]
        # Restrict the candidate to zones with the same source swatch where available.
        rgb = evidence.get("colorRgb")
        if rgb:
            hx = "#" + "".join(f"{int(c):02x}" for c in rgb)
            exact = [z for z in zones if str(z.get("hex", "")).lower() == hx.lower()]
            if exact:
                zones = exact
        views = []
        patch_meta = [{"viewIndex": i, "sourceImageId": view.get("imageId", view.get("image")), "rectPath": str(image_dir / view["rect"])}
                      for i, view in enumerate(side["views"][:2])]
        for vi, view in enumerate(side["views"][:2]):
            rect_path = image_dir / view["rect"]
            full = Image.open(rect_path).convert("RGB")
            candidates = []
            for zi, zone in enumerate(zones):
                if float(zone.get("x1", 0)) <= float(zone.get("x0", 0)) or float(zone.get("z1", 0)) <= float(zone.get("z0", 0)):
                    continue
                pix, box = crop_zone(full, view, float(side["heightM"]), zone)
                mask = exclusion_mask(pix.shape[:2], box, float(view["ppm"]), float(side["heightM"]), zone, reading)
                # Profile groups/openings are normalized wall rectangles and can
                # exclude windows missed by older source-reading schemas.
                objects = []
                for group in profile.get("groups", []):
                    region = group.get("region") if isinstance(group, dict) else None
                    if isinstance(region, (list, tuple)) and len(region) == 4:
                        objects.append({"x0": float(region[0]) * float(side["lengthM"]),
                                        "z0": float(region[1]) * float(side["heightM"]),
                                        "x1": float(region[2]) * float(side["lengthM"]),
                                        "z1": float(region[3]) * float(side["heightM"])})
                for opening in profile.get("openings", []):
                    rect = opening.get("rect") if isinstance(opening, dict) else None
                    if isinstance(rect, (list, tuple)) and len(rect) == 4:
                        objects.append({"x0": float(rect[0]) * float(side["lengthM"]),
                                        "z0": float(rect[1]) * float(side["heightM"]),
                                        "x1": float(rect[2]) * float(side["lengthM"]),
                                        "z1": float(rect[3]) * float(side["heightM"])})
                if objects:
                    from regen.appearance_patches import _cut_rectangles
                    mask = _cut_rectangles(mask, box, float(view["ppm"]), float(side["heightM"]), zone, objects)
                if mask.mean() < .25 or int(mask.sum()) < 800:
                    continue
                meas = measure_view(pix, mask, float(view["ppm"]))
                candidates.append((int(mask.sum()), zi, pix, mask, meas, zone, box))
            if not candidates:
                views.append({"accepted": False, "reason": "no adequately masked timber wall patch", "ppm": view.get("ppm")})
                continue
            # Use largest eligible zone so separate color/material bands are not combined.
            _, zi, pix, mask, meas, zone, box = max(candidates, key=lambda x: x[0])
            meas["zoneIndex"] = zi
            meas["zoneBoundsM"] = [zone.get(k) for k in ("x0", "x1", "z0", "z1")]
            meas["imageId"] = view.get("imageId", view.get("image"))
            meas["rectPath"] = str(rect_path)
            views.append(meas)
            views[-1]["_diagnostic"] = (pix, mask, view, rect_path)
            patch_meta[vi].update({"coverage": round(float(mask.mean()), 5), "validPixels": int(mask.sum()),
                                   "sourceImageId": view.get("imageId", view.get("image"))})
        agreement = compare_views(views[0], views[1]) if len(views) >= 2 else {"accepted": False}
        # Pitch means the vertical image axis: horizontal board courses recur by height.
        vertical = agreement.get("axes", {}).get("vertical_repeat_px", {})
        horizontal = agreement.get("axes", {}).get("horizontal_repeat_px", {})
        accepted = bool(vertical.get("accepted")) and not bool(horizontal.get("accepted"))
        # Keep diagnostic crops only for a cross-view near miss or a pre-typed
        # horizontal-course control; do not emit a giant contact sheet.
        near_miss = any(v.get("axes", {}).get("vertical_repeat_px", {}).get("accepted") for v in views if isinstance(v, dict))
        if near_miss or evidence.get("pattern") == "horizontal_courses":
            for vi, v in enumerate(views[:2]):
                diag = v.pop("_diagnostic", None)
                if not diag:
                    continue
                pix, mask, view, rect_path = diag
                stem = profile["id"].replace("/", "_") + f"__view{vi}"
                Image.fromarray(np.dstack([pix, np.where(mask, 255, 0).astype(np.uint8)]), "RGBA").save(out_dir / f"{stem}.png")
                preview = Image.fromarray(pix).convert("RGB")
                preview.thumbnail((420, 260))
                card = Image.new("RGB", (440, 300), "white")
                card.paste(preview, ((440-preview.width)//2, 24))
                ImageDraw.Draw(card).text((8, 5), f"{profile['id']} v{vi} {view.get('imageId', view.get('image'))}", fill="black")
                thumbs.append(card)
                if vi < len(patch_meta):
                    patch_meta[vi]["diagnosticFile"] = str(out_dir / f"{stem}.png")
        for v in views:
            v.pop("_diagnostic", None)
        records.append({"featureId": profile.get("match", {}).get("featureId"), "profileId": profile["id"],
                        "claimedPattern": evidence.get("pattern"),
                        "proposedPattern": "inferred_horizontal_periodic_boards" if accepted else None,
                        "sourceIds": profile.get("sourceIds", []), "readingPath": str(reading_path),
                        "accepted": accepted, "viewResults": views, "agreement": agreement,
                        "coursePitchM": vertical.get("period_m") if accepted else None,
                        "horizontalAxisControl": horizontal,
                        "patches": patch_meta,
                        "negativeControlReason": (None if accepted else "no cross-view vertical repeat with >=3/4 disjoint strips per view, >=6 px sampling, >=2.5 cycles, and <=12% view agreement, or horizontal-axis alternative also accepted")})
    if thumbs:
        cols = 4
        sheet = Image.new("RGB", (cols*440, ((len(thumbs)+cols-1)//cols)*300), "#dddddd")
        for i, card in enumerate(thumbs):
            sheet.paste(card, ((i % cols)*440, (i // cols)*300))
        sheet.save(out_dir / "contact-sheet.jpg", quality=90)
    output = {"schema": "horizontal-timber-module-audit-v2", "diagnosticOnly": True,
              "profileSource": str(profiles_path), "records": records,
              "candidateCount": len(records), "acceptedCount": sum(bool(x.get("accepted")) for x in records),
              "limitations": ["source readings and profile group/opening rectangles supply geometry masks; no semantic segmenter or detector rerun",
                              "periodicity is not proof of board morphology where unread windows, trim, or repeated architectural details remain",
                              "no moduleEvidence or profile material field has been changed"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--readings-dir", type=Path, required=True)
    parser.add_argument("--sides", type=Path, action="append", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    result = run(a.profiles, a.readings_dir, a.sides, a.image_dir, a.patch_dir, a.output)
    print(json.dumps({"candidateCount": result["candidateCount"], "acceptedCount": result["acceptedCount"],
                      "features": [{"featureId": r.get("featureId"), "accepted": r.get("accepted"), "coursePitchM": r.get("coursePitchM")} for r in result["records"]]}, indent=2))


if __name__ == "__main__":
    main()
