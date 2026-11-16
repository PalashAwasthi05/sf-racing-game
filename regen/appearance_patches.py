"""Bounded source-photo appearance experiment; never applies patches to a scene.

The first registered view is a patch source and the second is held out. Output
patches remain in rectified wall-frame pixels with explicit masks and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FAMILIES = ("brick", "render", "timber", "concrete", "stone", "stucco")
OUT_OF_PLANE_FIELDS = ("bays", "bands", "piers", "balconies", "porches", "recesses", "columns", "signs", "awnings", "roofs", "parapets", "railings", "fireEscapes", "hedges", "planting", "stairs", "accessRoutes", "reliefs", "ornaments", "cornices", "hoods", "surrounds", "friezes", "storeyTrims")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crop_zone(image: Image.Image, view: dict, wall_height: float, zone: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return zone RGB and corresponding rectified source pixels (wall z up)."""
    rgb = np.asarray(image.convert("RGB"))
    pad = int(view.get("rectPad", 0))
    ppm = float(view["ppm"])
    content = rgb[pad:rgb.shape[0] - pad if pad else None, pad:rgb.shape[1] - pad if pad else None]
    h, w = content.shape[:2]
    x0 = max(0, min(w, round(float(zone["x0"]) * ppm)))
    x1 = max(0, min(w, round(float(zone["x1"]) * ppm)))
    y0 = max(0, min(h, round((wall_height - float(zone["z1"])) * ppm)))
    y1 = max(0, min(h, round((wall_height - float(zone["z0"])) * ppm)))
    return content[y0:y1, x0:x1].copy(), np.array([x0, y0, x1, y1], dtype=np.int32)


def _cut_rectangles(valid: np.ndarray, box: np.ndarray, ppm: float, wall_height: float,
                    zone: dict, objects: list[dict]) -> np.ndarray:
    h, w = valid.shape
    for obj in objects:
        if not isinstance(obj, dict) or not all(k in obj for k in ("x0", "x1", "z0", "z1")):
            continue
        ix0 = max(float(zone["x0"]), float(obj["x0"]))
        ix1 = min(float(zone["x1"]), float(obj["x1"]))
        iz0 = max(float(zone["z0"]), float(obj["z0"]))
        iz1 = min(float(zone["z1"]), float(obj["z1"]))
        if ix1 <= ix0 or iz1 <= iz0:
            continue
        x0 = max(0, round((ix0 - float(zone["x0"])) * ppm))
        x1 = min(w, round((ix1 - float(zone["x0"])) * ppm))
        y0 = round((wall_height - iz1) * ppm) - int(box[1])
        y1 = round((wall_height - iz0) * ppm) - int(box[1])
        y0, y1 = max(y0, 0), min(y1, h)
        if y1 > y0 and x1 > x0:
            valid[y0:y1, x0:x1] = False
    return valid


def host_surface_mask(shape: tuple[int, int], box: np.ndarray, ppm: float, wall_height: float,
                      zone: dict, reading: dict, semantic_codes: np.ndarray | None = None) -> np.ndarray:
    """Wall-plane support mask; remove projections/recesses but keep in-plane windows as alignment features."""
    valid = np.ones(shape, dtype=bool)
    if semantic_codes is not None:
        if semantic_codes.shape != shape:
            raise ValueError("semantic mask must match rectified pixels")
        # Keep opening pixels for feature matching; reject signs, occluders and sky.
        valid &= ~np.isin(semantic_codes, (2, 3, 4))
    for field in OUT_OF_PLANE_FIELDS:
        value = reading.get(field, [])
        objects = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        _cut_rectangles(valid, box, ppm, wall_height, zone, objects)
    return valid


def exclusion_mask(shape: tuple[int, int], box: np.ndarray, ppm: float, wall_height: float,
                   zone: dict, reading: dict, semantic_codes: np.ndarray | None = None) -> np.ndarray:
    """Texture mask: host wall only, excluding openings, modeled projections and semantic occluders."""
    valid = host_surface_mask(shape, box, ppm, wall_height, zone, reading, semantic_codes)
    h, w = shape
    if semantic_codes is not None:
        valid &= semantic_codes == 0  # SegFormer retained/other class; windows may be semantic opening class.
    h, w = shape
    # Cut openings and whole bank extents: frame and sash pixels should not be trained as wall finish.
    opening_objects = []
    for collection in ("openings", "banks"):
        opening_objects.extend(reading.get(collection, []))
    other_zones = [item for item in reading.get("materials", []) if item is not zone and item != zone]
    return _cut_rectangles(valid, box, ppm, wall_height, zone, opening_objects + other_zones)


def normalize_median(rgb: np.ndarray, valid: np.ndarray, hex_color: str) -> tuple[np.ndarray, list[float]]:
    """Match the reading's flat sRGB swatch by one global channel scale; no spatial de-lighting."""
    if valid.sum() == 0:
        raise ValueError("no valid pixels")
    target = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    source = np.median(rgb[valid], axis=0)
    gain = np.divide(target, np.maximum(source, 1.), out=np.ones(3), where=source > 1.)
    return np.uint8(np.clip(rgb.astype(float) * gain[None, None, :], 0, 255)), gain.tolist()


def detector_valid(shape: tuple[int, int], boxes: list[tuple]) -> np.ndarray:
    """Exclude detected windows/doors/signs/awnings; retain the raw boxes in the report for QA."""
    from PIL import ImageDraw
    mask = Image.new("L", (shape[1], shape[0]), 255)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
        draw.rectangle((max(0, x0), max(0, y0), min(shape[1] - 1, x1), min(shape[0] - 1, y1)), fill=0)
    return np.asarray(mask) > 0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Small D65 sRGB to Lab conversion for comparable pixel residuals."""
    x = rgb.astype(np.float32) / 255.
    x = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    xyz = x @ np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]], np.float32).T
    xyz /= np.array([.95047, 1., 1.08883], np.float32)
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16. / 116.)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def residuals(source: np.ndarray, heldout: np.ndarray, valid: np.ndarray, flat_hex: str) -> dict:
    a, gain_a = normalize_median(source, valid, flat_hex)
    b, gain_b = normalize_median(heldout, valid, flat_hex)
    delta = np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)
    base = np.array([int(flat_hex[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    flat_delta = np.linalg.norm(srgb_to_lab(np.broadcast_to(base, b.shape)) - srgb_to_lab(b), axis=-1)
    lum_a = a @ np.array([.299, .587, .114], np.float32)
    lum_b = b @ np.array([.299, .587, .114], np.float32)
    hp_a = lum_a - gaussian_filter(lum_a, sigma=max(1., min(source.shape[:2]) / 40.))
    hp_b = lum_b - gaussian_filter(lum_b, sigma=max(1., min(source.shape[:2]) / 40.))
    xa, xb = hp_a[valid], hp_b[valid]
    corr = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 1e-6 and xb.std() > 1e-6 else None
    raw_luma_a = source @ np.array([.299, .587, .114], np.float32)
    raw_luma_b = heldout @ np.array([.299, .587, .114], np.float32)
    raw_hp_a = raw_luma_a - gaussian_filter(raw_luma_a, sigma=max(1., min(source.shape[:2]) / 40.))
    raw_hp_b = raw_luma_b - gaussian_filter(raw_luma_b, sigma=max(1., min(source.shape[:2]) / 40.))
    raw_xa, raw_xb = raw_hp_a[valid], raw_hp_b[valid]
    raw_corr = float(np.corrcoef(raw_xa, raw_xb)[0, 1]) if raw_xa.std() > 1e-6 and raw_xb.std() > 1e-6 else None
    raw_med_a, raw_med_b = np.median(source[valid], axis=0), np.median(heldout[valid], axis=0)
    raw_color_delta = float(np.linalg.norm(srgb_to_lab(raw_med_a[None, None, :]) - srgb_to_lab(raw_med_b[None, None, :]), axis=-1)[0, 0])
    return {"validPixels": int(valid.sum()), "validFraction": float(valid.mean()),
            "sourceRawMedianRgb": [round(float(x), 2) for x in raw_med_a], "heldoutRawMedianRgb": [round(float(x), 2) for x in raw_med_b],
            "sourceVsHeldoutRawMedianDeltaE76": raw_color_delta,
            "sourceMedianGainToSwatch": gain_a, "heldoutMedianGainToSwatch": gain_b,
            "swatchNormalizedPatchVsHeldoutMedianDeltaE76": float(np.median(delta[valid])),
            "heldoutFlatSwatchVsPhotoMedianDeltaE76": float(np.median(flat_delta[valid])),
            "swatchNormalizedSourceTextureHighpassStdSrgbLuma": float(xa.std()), "swatchNormalizedHeldoutTextureHighpassStdSrgbLuma": float(xb.std()),
            "swatchNormalizedTextureHighpassCorrelation": corr,
            "rawSourceTextureHighpassStdSrgbLuma": float(raw_xa.std()), "rawHeldoutTextureHighpassStdSrgbLuma": float(raw_xb.std()),
            "rawTextureHighpassRmseSrgbLuma": float(np.sqrt(np.mean((raw_xa - raw_xb) ** 2))),
            "rawTextureHighpassCorrelation": raw_corr,
            "interpretation": "raw color and high-pass comparisons do not use the reading hex. Swatch-normalized residuals are circular with respect to color because both views are independently scaled to that hex. Global gains remove only per-view color/brightness scale; spatial shadows/speculars remain; residuals are not albedo or physical relief"}, a, b


def _zone_candidates(readings_dir: Path, sides_dir: Path) -> list[tuple[dict, dict, dict, Path]]:
    side_by_id = {}
    for file in sorted(sides_dir.glob("sides*.json")):
        try:
            for piece in json.loads(file.read_text()).get("pieces", []):
                if len(piece.get("views", [])) >= 2 and all(v.get("registered") == "skyline" for v in piece["views"][:2]):
                    side_by_id[piece["id"]] = piece
        except (OSError, ValueError):
            continue
    result = []
    for file in sorted(readings_dir.glob("*.json")):
        try:
            reading = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        piece = side_by_id.get(reading.get("id"))
        if not piece or not reading.get("registeredCrops"):
            continue
        for zone in reading.get("materials", []):
            family = str(zone.get("family", "")).lower()
            if family not in FAMILIES or not isinstance(zone.get("hex"), str) or len(zone["hex"]) != 7:
                continue
            x0, x1 = float(zone.get("x0", 0)), float(zone.get("x1", 0))
            z0, z1 = float(zone.get("z0", 0)), float(zone.get("z1", 0))
            area = max(0., x1 - x0) * max(0., z1 - z0)
            ppm = min(float(v.get("ppm", 0)) for v in piece["views"][:2])
            # Generic minimum source density/area; ranking prefers larger, better sampled opaque patches.
            if area < 4.0 or ppm < 12 or z0 < 0 or z1 > float(piece["heightM"]):
                continue
            result.append((reading, piece, zone, file))
    return result


def _render_candidate(candidate, crop_dir: Path, segmenter, detector, out_dir: Path, source_images_dir: Path | None = None) -> dict:
    reading, piece, zone, reading_path = candidate
    views = piece["views"][:2]
    crops = []
    boxes = []
    for view in views:
        path = crop_dir / view["rect"]
        full = Image.open(path).convert("RGB")
        pixels, box = crop_zone(full, view, float(piece["heightM"]), zone)
        boxes.append(box)
        crops.append((path, pixels, box))
    if crops[0][1].shape != crops[1][1].shape or not crops[0][1].size:
        raise ValueError("registered view zone dimensions do not agree")
    semantic = [segmenter.labels(x[1]) for x in crops] if segmenter else [None, None]
    masks = [exclusion_mask(x[1].shape[:2], x[2], float(views[i]["ppm"]), float(piece["heightM"]), zone, reading, semantic[i]) for i, x in enumerate(crops)]
    geometry_masks = [exclusion_mask(x[1].shape[:2], x[2], float(views[i]["ppm"]), float(piece["heightM"]), zone, reading) for i, x in enumerate(crops)]
    host_masks = [host_surface_mask(x[1].shape[:2], x[2], float(views[i]["ppm"]), float(piece["heightM"]), zone, reading, semantic[i]) for i, x in enumerate(crops)]
    detected_boxes = [detector.boxes(x[1]) for x in crops] if detector else [[], []]
    detection_masks = [detector_valid(x[1].shape[:2], detected_boxes[i]) for i, x in enumerate(crops)]
    masks = [masks[i] & detection_masks[i] for i in range(2)]
    from scipy.ndimage import binary_erosion
    masks = [binary_erosion(mask, iterations=1, border_value=0) for mask in masks]
    source_valid, heldout_valid = masks
    from regen.align_surface_photos import align as align_surface_photos
    alignment = align_surface_photos(crops[0][1], crops[1][1], host_masks[0], host_masks[1])
    source_aligned = None
    shared_valid = None
    if alignment.get("accepted"):
        import cv2
        H = np.asarray(alignment["homographySourceToTarget"], dtype=float)
        h, w = crops[1][1].shape[:2]
        source_aligned = cv2.warpPerspective(crops[0][1], H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        source_valid_aligned = cv2.warpPerspective(source_valid.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        shared_valid = binary_erosion(source_valid_aligned & heldout_valid, iterations=1, border_value=0)
    patch_a, gain_a = normalize_median(crops[0][1], source_valid, zone["hex"])
    heldout_preview, gain_b = normalize_median(crops[1][1], heldout_valid, zone["hex"])
    metrics = {"sourceValidPixels": int(source_valid.sum()), "sourceValidFraction": float(source_valid.mean()),
               "heldoutValidPixels": int(heldout_valid.sum()), "heldoutValidFraction": float(heldout_valid.mean()),
               "sourceMedianGainToSwatch": gain_a, "heldoutMedianGainToSwatch": gain_b}
    source_aligned_normalized = None
    if shared_valid is not None and shared_valid.sum() >= 200:
        heldout_metrics, source_aligned_normalized, heldout_normalized = residuals(source_aligned, crops[1][1], shared_valid, zone["hex"])
        metrics.update({"sharedValidPixelsAfterAlignment": int(shared_valid.sum()), "sharedValidFractionInHeldoutFrame": float(shared_valid.mean()), **heldout_metrics})
    else:
        heldout_normalized = heldout_preview
        metrics.update({"sharedValidPixelsAfterAlignment": 0, "sharedValidFractionInHeldoutFrame": 0.0,
                        "heldoutMetricsStatus": "not computed: independent feature alignment failed or left fewer than 200 shared valid pixels"})
    metrics["coverageGate25Percent"] = "pass" if metrics["sharedValidFractionInHeldoutFrame"] >= .25 else "fail"
    metrics["sourceCoverageGate25Percent"] = "pass" if metrics["sourceValidFraction"] >= .25 else "fail"
    metrics["interpretation"] = "Raw cross-view texture metrics use SIFT/RANSAC-aligned pixels and no reading hex. Swatch-normalized residuals are circular because each view is independently scaled to that hex. Homography acceptance is image-space only: absolute camera position and physical plane remain unvalidated. No albedo/normal/depth inference."
    if source_valid.sum() < 200:
        raise ValueError(f"insufficient source A mask coverage for a diagnostic patch ({source_valid.sum()} px)")
    family = str(zone["family"]).lower()
    slug = reading["id"].replace("/", "_") + "__" + family
    alpha = np.where(source_valid, 255, 0).astype(np.uint8)
    alpha_b = np.where(heldout_valid, 255, 0).astype(np.uint8)
    alpha_shared = np.where(shared_valid, 255, 0).astype(np.uint8) if shared_valid is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([patch_a, alpha]), "RGBA").save(out_dir / f"{slug}__source-a-patch.png")
    Image.fromarray(alpha, "L").save(out_dir / f"{slug}__valid-mask.png")
    Image.fromarray(alpha_b, "L").save(out_dir / f"{slug}__heldout-valid-mask.png")
    Image.fromarray(np.where(geometry_masks[0], 255, 0).astype(np.uint8), "L").save(out_dir / f"{slug}__geometry-valid-mask.png")
    Image.fromarray(np.where(geometry_masks[1], 255, 0).astype(np.uint8), "L").save(out_dir / f"{slug}__geometry-valid-mask-b.png")
    Image.fromarray(np.where(host_masks[0], 255, 0).astype(np.uint8), "L").save(out_dir / f"{slug}__host-surface-mask-a.png")
    Image.fromarray(np.where(host_masks[1], 255, 0).astype(np.uint8), "L").save(out_dir / f"{slug}__host-surface-mask-b.png")
    for i in range(2):
        Image.fromarray(np.where(detection_masks[i], 255, 0).astype(np.uint8), "L").save(out_dir / f"{slug}__detector-valid-mask-{chr(ord('a') + i)}.png")
    if semantic[0] is not None:
        Image.fromarray(semantic[0].astype(np.uint8), "L").save(out_dir / f"{slug}__semantic-labels-a.png")
        Image.fromarray(semantic[1].astype(np.uint8), "L").save(out_dir / f"{slug}__semantic-labels-b.png")
    Image.fromarray(np.dstack([heldout_normalized, alpha_shared if alpha_shared is not None else alpha_b]), "RGBA").save(out_dir / f"{slug}__heldout-b-normalized.png")
    Image.fromarray(np.dstack([crops[0][1], alpha]), "RGBA").save(out_dir / f"{slug}__source-a-raw.png")
    if source_aligned_normalized is not None and alpha_shared is not None:
        Image.fromarray(np.dstack([source_aligned_normalized, alpha_shared]), "RGBA").save(out_dir / f"{slug}__source-a-aligned-to-b.png")
    def view_record(i):
        view = views[i]
        pano = source_images_dir / f"{view['image']}.jpg" if source_images_dir is not None else None
        return {"imageId": view["image"], "captureDate": view["captureDate"], "distanceM": view["distanceM"],
                "rect": str(crops[i][0]), "rectSha256": sha256(crops[i][0]), "cropBoxPx": boxes[i].tolist(),
                "sourcePanorama": str(pano) if pano is not None and pano.is_file() else None, "sourcePanoramaSha256": sha256(pano) if pano is not None and pano.is_file() else None}
    return {"readingId": reading["id"], "family": family, "zone": {k: zone.get(k) for k in ("x0", "x1", "z0", "z1", "hex", "pattern")},
            "readingFile": str(reading_path), "readingSha256": sha256(reading_path), "wallFrame": {k: piece.get(k) for k in ("frameU", "lengthM", "heightM", "normal")},
            "views": [view_record(i) for i in range(len(views))],
            "maskSource": "SegFormer + Grounding DINO + measured openings and out-of-plane wall assemblies" if segmenter else "detector/opening and out-of-plane wall assembly masks",
            "hostSurfaceMaskCoverage": [float(x.mean()) for x in host_masks], "alignment": alignment,
            "detectorSource": "cached local Grounding DINO boxes for windows/doors/signs/awnings" if detector else "disabled",
            "detectedBoxes": [[{"xyxy": [round(float(v), 1) for v in b[:4]], "kind": b[4], "score": round(float(b[5]), 4)} for b in boxes] for boxes in detected_boxes],
            "semanticLabelCodes": {"0": "other/retained", "1": "opening excluded", "2": "sign excluded", "3": "occluder excluded", "4": "sky excluded"} if segmenter else None,
            "coverageGate25Percent": metrics["coverageGate25Percent"],
            "patchStatus": "diagnostic-only; no scene application; image homography does not establish that every included pixel belongs to one physical plane",
            "metrics": metrics, "outputs": [f"{slug}__source-a-patch.png", f"{slug}__valid-mask.png", f"{slug}__heldout-valid-mask.png", f"{slug}__geometry-valid-mask.png", f"{slug}__geometry-valid-mask-b.png", f"{slug}__host-surface-mask-a.png", f"{slug}__host-surface-mask-b.png", f"{slug}__detector-valid-mask-a.png", f"{slug}__detector-valid-mask-b.png", f"{slug}__heldout-b-normalized.png", f"{slug}__source-a-raw.png"] + ([f"{slug}__source-a-aligned-to-b.png"] if source_aligned_normalized is not None else []) + ([f"{slug}__semantic-labels-a.png", f"{slug}__semantic-labels-b.png"] if semantic[0] is not None else [])}


def run(readings_dir: Path, sides_dir: Path, crop_dir: Path, output_dir: Path, source_images_dir: Path | None = None,
        families: tuple[str, ...] = ("brick", "render"), use_segmentation: bool = True, use_detector: bool = True) -> dict:
    candidates = _zone_candidates(readings_dir, sides_dir)
    # Rank generically within each requested family by observed area x registered source pixel density.
    chosen = []
    for family in families:
        pool = [c for c in candidates if str(c[2].get("family", "")).lower() == family]
        pool.sort(key=lambda c: -((float(c[2]["x1"]) - float(c[2]["x0"])) * (float(c[2]["z1"]) - float(c[2]["z0"])) * min(float(v["ppm"]) for v in c[1]["views"][:2])))
        if pool:
            chosen.append(pool[0])
    if len(chosen) < 2:
        raise RuntimeError(f"need two distinct material families with paired registered crops; found {[c[2].get('family') for c in chosen]}")
    segmenter = None
    if use_segmentation:
        from facade_segment_v27 import Segmenter
        segmenter = Segmenter()
    detector = None
    if use_detector:
        from facade_segment_v27 import Detector
        detector = Detector()
    records = []
    for c in chosen:
        try:
            records.append(_render_candidate(c, crop_dir, segmenter, detector, output_dir, source_images_dir))
        except Exception as exc:
            records.append({"readingId": c[0]["id"], "family": c[2].get("family"), "status": "rejected", "reason": f"{type(exc).__name__}: {exc}"})
    successful = [r for r in records if "metrics" in r]
    for record in successful:
        record["outputSha256"] = {name: sha256(output_dir / name) for name in record["outputs"] if (output_dir / name).is_file()}
    if successful:
        _contact_sheet(successful, output_dir)
    report = {"schema": "appearance-patch-experiment-r25", "replay": "geoenv/bin/python regen/appearance_patches.py --readings areas/review/facade-rethink/readings-aperture-access-r25 --sides areas/pacific-heights/regen/sides-reg --crops areas/pacific-heights/regen/sides-reg/img --source-images areas/pacific-heights/imagery/images --output areas/review/facade-rethink/appearance-patches-r25 --families brick,render", "experimentScriptSha256": sha256(Path(__file__)), "appliedToScene": False,
              "segmentation": "local facade_segment_v27 SegFormer-B2; its other code is retained and opening/sign/occluder/sky codes excluded" if segmenter else "disabled",
              "detector": "local facade_segment_v27 Grounding DINO; window/door/sign/awning boxes excluded" if detector else "disabled",
              "records": records, "summary": {"selected": len(records), "diagnosticOutputs": len(successful), "coverageGatePassesAt25Percent": sum(r.get("coverageGate25Percent") == "pass" for r in successful), "coverageGateFailuresAt25Percent": sum(r.get("coverageGate25Percent") == "fail" for r in successful), "rejectedBelow12PercentOr200Pixels": len(records) - len(successful)},
              "limitations": "Diagnostic output requires at least 200 valid source-A pixels. The original 25% shared coverage gate is preserved and reported. Two source views per zone only; feature alignment is image-space evidence, not physical plane proof. Global channel median matching is not spatial de-lighting; patches encode visible RGB, not albedo, normals, depth, or certified texture. Masks need visual QA; no scene receives these patches."}
    if (output_dir / "contact-sheet.jpg").is_file():
        report["contactSheetSha256"] = sha256(output_dir / "contact-sheet.jpg")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _contact_sheet(records: list[dict], output: Path) -> None:
    tiles = []
    for rec in records:
        stem = rec["outputs"][0].replace("__source-a-patch.png", "")
        panels = [_composite_masked(Image.open(output / f"{stem}__source-a-raw.png")),
                  Image.open(output / f"{stem}__valid-mask.png").convert("RGB"),
                  _composite_masked(Image.open(output / f"{stem}__source-a-patch.png")),
                  _composite_masked(Image.open(output / f"{stem}__heldout-b-normalized.png"))]
        width = 320
        thumbs = [im.resize((width, max(1, round(im.height * width / im.width)))) for im in panels]
        h = max(im.height for im in thumbs)
        row = Image.new("RGB", (width * 4, h + 40), "white")
        draw = ImageDraw.Draw(row)
        for i, im in enumerate(thumbs):
            row.paste(im, (i * width, 40))
            draw.text((i * width + 4, 6), ["source A", "valid mask", "A normalized to swatch", "held-out B normalized"][i], fill="black")
        tiles.append(row)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y)); y += tile.height
    sheet.save(output / "contact-sheet.jpg", quality=92)


def _composite_masked(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (176, 176, 176, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--sides", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, help="optional folder for raw panorama source hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default="brick,render")
    parser.add_argument("--no-segmentation", action="store_true")
    parser.add_argument("--no-detection", action="store_true")
    args = parser.parse_args()
    report = run(args.readings, args.sides, args.crops, args.output, args.source_images, tuple(x.strip().lower() for x in args.families.split(",") if x.strip()), not args.no_segmentation, not args.no_detection)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
