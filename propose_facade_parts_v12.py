"""Grounding DINO proposal diagnostics for rectified facade photographs.

Detections from this command are visual proposals only.  They are retained with
their source crop and alpha-validity provenance, but never become facade
openings, measurements, textures, or world admissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_CACHE_DIR = Path("work/.models")
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.20
DEFAULT_ALPHA_MINIMUM = 0.25
DEFAULT_LIMIT = 24
PREPROCESS_VERSION = "grounding-dino-proposals-v2-alpha-padding32"
PROMPT_LABELS = ("window", "door", "sign", "balcony")
PROMPT = ". ".join(PROMPT_LABELS) + "."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def canonical_label(value: Any) -> str:
    label = _normal_label(value)
    for category in PROMPT_LABELS:
        variants = {category, category + "s"}
        if category == "balcony":
            variants.add("balconies")
        if any(re.search(rf"\b{re.escape(variant)}\b", label) for variant in variants):
            return category
    return "unknown"


def clip_box(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """Return a finite, positive box clipped to image bounds."""

    values = np.asarray(list(box), dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x0, y0, x1, y1 = values
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    left, right = np.clip((left, right), 0.0, float(width))
    top, bottom = np.clip((top, bottom), 0.0, float(height))
    if right <= left or bottom <= top:
        return None
    return float(left), float(top), float(right), float(bottom)


def alpha_overlap(alpha: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Measure the fraction of a proposal box with nonzero crop validity."""

    if alpha.ndim != 2:
        raise ValueError("alpha must be a 2-D array")
    height, width = alpha.shape
    x0, y0, x1, y1 = box
    left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    if right <= left or bottom <= top:
        return 0.0
    region = alpha[top:bottom, left:right]
    return float(np.mean(region > 0)) if region.size else 0.0


def _false_positive_reasons(box: tuple[float, float, float, float], width: int, height: int, alpha_fraction: float) -> list[str]:
    x0, y0, x1, y1 = box
    reasons = []
    if x0 <= 0.0 or y0 <= 0.0 or x1 >= width or y1 >= height:
        reasons.append("touches_crop_edge")
    if (x1 - x0) * (y1 - y0) < max(16.0, width * height * 0.001):
        reasons.append("very_small_box")
    if alpha_fraction < 0.50:
        reasons.append("low_alpha_validity_fraction")
    return reasons


def filter_detections(
    boxes: Iterable[Iterable[float]],
    scores: Iterable[float],
    labels: Iterable[Any],
    alpha: np.ndarray,
    *,
    alpha_minimum: float = DEFAULT_ALPHA_MINIMUM,
) -> list[dict[str, Any]]:
    """Clip model detections and reject proposals on invalid crop padding."""

    if not 0.0 <= alpha_minimum <= 1.0:
        raise ValueError("alpha_minimum must be between zero and one")
    height, width = alpha.shape
    output: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        clipped = clip_box(box, width, height)
        if clipped is None or not np.isfinite(score_value):
            continue
        fraction = alpha_overlap(alpha, clipped)
        if fraction < alpha_minimum:
            continue
        category = canonical_label(label)
        reasons = _false_positive_reasons(clipped, width, height, fraction)
        if score_value < 0.30:
            reasons.append("low_model_score")
        output.append(
            {
                "box": [float(value) for value in clipped],
                "score": score_value,
                "label": str(label),
                "category": category,
                "alphaValidityFraction": fraction,
                "likelyFalsePositiveReasons": reasons,
            }
        )
    return output


def _load_model(model_id: str, cache_dir: Path, device: torch.device) -> tuple[Any, Any, Path]:
    from huggingface_hub import snapshot_download

    model_dir = Path(
        snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            allow_patterns=["config.json", "preprocessor_config.json", "tokenizer*", "vocab*", "merges*", "*.json", "*.txt", "*.model", "*.safetensors"],
        )
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir,
        trust_remote_code=False,
        use_safetensors=True,
        weights_only=True,
    )
    model.eval().to(device)
    return processor, model, model_dir


def _device(name: str) -> torch.device:
    if name != "cpu" and name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_hash(model_dir: Path) -> str:
    files = sorted(path for path in model_dir.rglob("*") if path.is_file() and path.stat().st_size > 0)
    return hashlib.sha256("".join(f"{path.name}:{_sha256(path)}" for path in files).encode()).hexdigest()


def _cache_key(crop_sha256: str, model_hash: str, *, box_threshold: float, text_threshold: float, alpha_minimum: float, device: torch.device) -> str:
    payload = {
        "cropSha256": crop_sha256,
        "modelHash": model_hash,
        "boxThreshold": float(box_threshold),
        "textThreshold": float(text_threshold),
        "alphaMinimum": float(alpha_minimum),
        "prompt": PROMPT,
        "preprocessVersion": PREPROCESS_VERSION,
        "device": str(device),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def infer_proposals(
    image: Image.Image,
    processor: Any,
    model: Any,
    device: torch.device,
    *,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    alpha_minimum: float = DEFAULT_ALPHA_MINIMUM,
) -> list[dict[str, Any]]:
    """Run Grounding DINO and return alpha-valid visual proposals."""

    rgba = image.convert("RGBA")
    original_alpha = np.asarray(rgba)[..., 3]
    # Grounding DINO's feature map cannot reliably handle a one-pixel-high
    # diagnostic sliver.  Add transparent padding only for inference, then
    # clip proposals back to the original crop and recheck original alpha.
    padded = rgba
    minimum_side = 32
    if rgba.width < minimum_side or rgba.height < minimum_side:
        padded = Image.new("RGBA", (max(rgba.width, minimum_side), max(rgba.height, minimum_side)), (0, 0, 0, 0))
        padded.paste(rgba, (0, 0))
    alpha = np.asarray(padded)[..., 3]
    inputs = processor(images=padded.convert("RGB"), text=[PROMPT], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.get("input_ids"),
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(padded.height, padded.width)],
        text_labels=[list(PROMPT_LABELS)],
    )[0]
    boxes = result["boxes"].detach().cpu().numpy() if hasattr(result["boxes"], "detach") else np.asarray(result["boxes"])
    scores = result["scores"].detach().cpu().numpy() if hasattr(result["scores"], "detach") else np.asarray(result["scores"])
    labels = result.get("text_labels", result.get("labels", []))
    proposals = filter_detections(boxes, scores, labels, alpha, alpha_minimum=alpha_minimum)
    if padded.size == rgba.size:
        return proposals
    clipped: list[dict[str, Any]] = []
    for proposal in proposals:
        box = clip_box(proposal["box"], rgba.width, rgba.height)
        if box is None:
            continue
        fraction = alpha_overlap(original_alpha, box)
        if fraction < alpha_minimum:
            continue
        updated = dict(proposal)
        updated["box"] = [float(value) for value in box]
        updated["alphaValidityFraction"] = fraction
        reasons = _false_positive_reasons(box, rgba.width, rgba.height, fraction)
        if updated["score"] < 0.30:
            reasons.append("low_model_score")
        updated["likelyFalsePositiveReasons"] = reasons
        clipped.append(updated)
    return clipped


def _color(category: str) -> tuple[int, int, int]:
    palette = {"window": (55, 180, 255), "door": (80, 220, 100), "sign": (255, 180, 45), "balcony": (220, 90, 235), "unknown": (240, 240, 240)}
    return palette.get(category, palette["unknown"])


def render_proposals(image: Image.Image, proposals: list[dict[str, Any]]) -> Image.Image:
    rendered = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(rendered)
    for proposal in proposals:
        x0, y0, x1, y1 = proposal["box"]
        color = _color(proposal["category"])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=max(1, round(min(rendered.size) / 320)))
        label = f"{proposal['category']} {proposal['score']:.2f}"
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)
    return rendered


def _contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    tile = 300
    columns = min(4, len(paths))
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 22)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile, tile))
            x = (index % columns) * tile + (tile - image.width) // 2
            y = (index // columns) * (tile + 22) + (tile - image.height) // 2
            sheet.paste(image, (x, y))
        draw.text(((index % columns) * tile + 4, (index // columns) * (tile + 22) + tile + 3), path.stem, fill="white")
    sheet.save(output)


def run(
    input_path: Path,
    output: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    alpha_minimum: float = DEFAULT_ALPHA_MINIMUM,
    device_name: str = "cpu",
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Run bounded proposal diagnostics over crop provenance."""

    if limit < 1:
        raise ValueError("limit must be positive")
    started = time.perf_counter()
    provenance = json.loads(input_path.read_text())
    crops = list(provenance.get("crops", []))[:limit]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = _device(device_name)
    processor, model, model_dir = _load_model(model_id, cache_dir, device)
    model_hash = _model_hash(model_dir)
    cache_dir_output = output / "proposal-cache"
    cache_dir_output.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0
    records: list[dict[str, Any]] = []
    rendered_paths: list[Path] = []
    paired_views = bool(provenance.get("selection", {}).get("pairedIndependentViews"))
    occurrences: dict[str, int] = defaultdict(int)
    for crop in crops:
        building_key = str(crop.get("buildingKey", "unknown"))
        occurrence = occurrences[building_key]
        occurrences[building_key] += 1
        explicit_pair = crop.get("pairId") or crop.get("pairKey")
        frame_key = crop.get("frontageFrameIndex")
        if frame_key is None and isinstance(crop.get("metricFrame"), dict):
            frame_key = crop["metricFrame"].get("frameIndex") or crop["metricFrame"].get("id")
        pair_key = str(explicit_pair) if explicit_pair is not None else (f"{building_key}::frame-{frame_key}" if paired_views and frame_key is not None else None)
        item: dict[str, Any] = {
            "index": crop.get("index"),
            "sourceId": crop.get("sourceId"),
            "captureDate": crop.get("captureDate"),
            "sourceUrl": crop.get("sourceUrl"),
            "cropPath": crop.get("path"),
            "validMaskPath": crop.get("validMaskPath"),
            "inputOcclusionUnchecked": crop.get("occlusionUnchecked", True),
            "screenVisibilityGeometryOnly": crop.get("screenVisibilityGeometryOnly", True),
            "worldRegistrationValidated": False,
            "facadeMeasurementsGenerated": 0,
            "texturesPublished": False,
            "status": "rejected",
            "proposalAdmission": False,
            "pairKey": pair_key,
            "pairViewIndex": (occurrence % 2) if pair_key is not None else None,
        }
        try:
            crop_path = Path(crop["path"])
            crop_sha256 = _sha256(crop_path)
            cache_path = cache_dir_output / f"crop-{int(crop.get('index', len(records))):03d}.json"
            cache_key = _cache_key(crop_sha256, model_hash, box_threshold=box_threshold, text_threshold=text_threshold, alpha_minimum=alpha_minimum, device=device)
            cached = None
            if cache_path.exists():
                try:
                    candidate_cache = json.loads(cache_path.read_text())
                    if candidate_cache.get("status") == "diagnostic" and candidate_cache.get("cacheKey") == cache_key and candidate_cache.get("cropSha256") == crop_sha256:
                        cached = candidate_cache
                except (OSError, ValueError, TypeError):
                    cached = None
            with Image.open(crop_path) as source:
                image = source.convert("RGBA")
            if cached is not None:
                proposals = list(cached.get("proposals", []))
                cache_hits += 1
            else:
                cache_misses += 1
                proposals = infer_proposals(image, processor, model, device, box_threshold=box_threshold, text_threshold=text_threshold, alpha_minimum=alpha_minimum)
                cache_path.write_text(json.dumps({"schema": "facade-part-proposal-cache/v1", "status": "diagnostic", "cacheKey": cache_key, "cropSha256": crop_sha256, "modelHash": model_hash, "modelId": model_id, "prompt": PROMPT, "boxThreshold": box_threshold, "textThreshold": text_threshold, "alphaMinimum": alpha_minimum, "preprocessVersion": PREPROCESS_VERSION, "device": str(device), "proposals": proposals}, indent=2))
                cache_writes += 1
            stem = f"crop-{int(crop.get('index', len(records))):03d}"
            rendered_path = output / f"{stem}-proposals.png"
            render_proposals(image, proposals).save(rendered_path)
            rendered_paths.append(rendered_path)
            item.update({"status": "diagnostic", "proposals": proposals, "proposalCount": len(proposals), "renderedPath": str(rendered_path.resolve()), "alphaMinimum": alpha_minimum, "cachePath": str(cache_path.resolve()), "cacheReused": cached is not None, "cropSha256": crop_sha256})
        except Exception as exc:
            item["reason"] = f"crop failed: {type(exc).__name__}: {exc}"
        records.append(item)
    contact_sheet = output / "contact-sheet-proposals.png"
    _contact_sheet(rendered_paths, contact_sheet)
    detections = [proposal for item in records if item["status"] == "diagnostic" for proposal in item.get("proposals", [])]
    category_counts: dict[str, int] = {}
    likely_false_positive_counts: dict[str, int] = {}
    detections_by_pair: dict[str, int] = defaultdict(int)
    for proposal in detections:
        category_counts[proposal["category"]] = category_counts.get(proposal["category"], 0) + 1
        for reason in proposal["likelyFalsePositiveReasons"]:
            likely_false_positive_counts[reason] = likely_false_positive_counts.get(reason, 0) + 1
    for item in records:
        if item.get("pairKey") is not None:
            detections_by_pair[item["pairKey"]] += int(item.get("proposalCount", 0))
    report = {
        "schema": "facade-part-proposal-diagnostic/v1",
        "worldRegistrationValidated": False,
        "facadeMeasurementsGenerated": 0,
        "texturesPublished": False,
        "proposalAdmission": False,
        "evidenceScope": "Grounding DINO visual proposals only; no opening geometry, measurement, occlusion truth, registration, or admission claim",
        "input": str(input_path.resolve()),
        "inputSha256": _sha256(input_path),
        "model": {"id": model_id, "resolvedDirectory": str(model_dir.resolve()), "hash": _model_hash(model_dir), "prompt": PROMPT, "labels": list(PROMPT_LABELS), "boxThreshold": box_threshold, "textThreshold": text_threshold, "device": str(device), "torchThreads": 1},
        "alphaPolicy": "A proposal is retained only when its clipped box overlaps at least the configured fraction of nonzero crop alpha; alpha does not certify surface correctness or remove real-world occlusion.",
        "contactSheet": str(contact_sheet.resolve()) if rendered_paths else None,
        "runtime": {"elapsedSeconds": float(time.perf_counter() - started), "device": str(device), "torchThreads": int(torch.get_num_threads()), "torchInteropThreads": 1},
        "cache": {"directory": str(cache_dir_output.resolve()), "preprocessVersion": PREPROCESS_VERSION, "modelHash": model_hash, "hits": cache_hits, "misses": cache_misses, "writes": cache_writes},
        "crops": records,
        "summary": {"requested": len(crops), "diagnosticCompleted": sum(item["status"] == "diagnostic" for item in records), "failed": sum(item["status"] == "rejected" for item in records), "actualDetections": len(detections), "detectionsByCategory": category_counts, "detectionsByPair": dict(sorted(detections_by_pair.items())), "likelyFalsePositiveReasons": likely_false_positive_counts},
        "limitations": ["Grounding DINO boxes are candidate regions only and may identify signs, trees, cars, shadows, or other lookalikes.", "No segmentation, geometry registration, or source occlusion truth was applied.", "No candidate is admitted as a window, door, sign, balcony, or facade measurement."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--alpha-minimum", type=float, default=DEFAULT_ALPHA_MINIMUM)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    report = run(args.input, args.output, model_id=args.model_id, cache_dir=args.cache_dir, box_threshold=args.box_threshold, text_threshold=args.text_threshold, alpha_minimum=args.alpha_minimum, device_name=args.device, limit=args.limit)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
