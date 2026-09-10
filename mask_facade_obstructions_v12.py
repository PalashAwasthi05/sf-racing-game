"""Diagnostic semantic masking for rectified facade crops.

The result is an obstruction hypothesis, not a facade measurement or an
opening mask.  Class decisions come from the selected model's ``id2label``
configuration; no model class numbers or site-specific rules are embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DEFAULT_CACHE_DIR = Path("work/.models")
INFERENCE_CAP = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_label(label: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def label_groups(id2label: dict[Any, Any]) -> dict[str, list[int]]:
    """Map configured labels to semantic groups without numeric assumptions."""

    groups = {"keep": [], "exclude": [], "openingCandidate": [], "unknown": []}
    keep_terms = {"building", "wall", "house"}
    opening_terms = {"window", "windowpane", "door", "screen door"}
    exclude_terms = {"vegetation", "tree", "plant", "grass", "sky", "car", "truck", "bus", "train", "person", "rider", "bicycle", "motorcycle", "pole", "traffic light", "traffic sign", "road", "sidewalk", "terrain", "earth", "floor", "ceiling"}
    for raw_id, raw_label in id2label.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        label = _normal_label(raw_label)
        if label in keep_terms:
            groups["keep"].append(class_id)
        elif label in opening_terms:
            groups["openingCandidate"].append(class_id)
        elif label in exclude_terms:
            groups["exclude"].append(class_id)
        else:
            groups["unknown"].append(class_id)
    return groups


def resize_pad(image: Image.Image, cap: int = INFERENCE_CAP) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Resize without stretching and pad to the model input square."""

    width, height = image.size
    scale = min(cap / max(width, height), 1.0)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (cap, cap), (0, 0, 0))
    left = (cap - resized.width) // 2
    top = (cap - resized.height) // 2
    canvas.paste(resized.convert("RGB"), (left, top))
    return canvas, (left, top, left + resized.width, top + resized.height), scale


def _probability_png(probabilities: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(probabilities * 255.0), 0, 255).astype(np.uint8), mode="L")


def infer_mask(image: Image.Image, processor: Any, model: Any, groups: dict[str, list[int]], threshold: float, device: torch.device) -> dict[str, Any]:
    original = image.convert("RGBA")
    padded, box, scale = resize_pad(original)
    inputs = processor(images=padded, return_tensors="pt", do_resize=False)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values).logits
    logits = torch.nn.functional.interpolate(logits, size=(INFERENCE_CAP, INFERENCE_CAP), mode="bilinear", align_corners=False)[0]
    probability = torch.softmax(logits, dim=0).detach().cpu().numpy()
    left, top, right, bottom = box
    crop = probability[:, top:bottom, left:right]
    resized = np.stack([np.asarray(Image.fromarray(channel).resize(original.size, Image.Resampling.BILINEAR)) for channel in crop], axis=0)
    # Bilinear probabilities can drift slightly; renormalize before argmax.
    resized = np.maximum(resized, 0.0)
    resized /= np.maximum(resized.sum(axis=0, keepdims=True), 1e-8)
    classes = np.argmax(resized, axis=0).astype(np.int32)
    confidence = np.max(resized, axis=0)
    confident = confidence >= threshold
    keep = confident & np.isin(classes, groups["keep"])
    excluded = confident & np.isin(classes, groups["exclude"])
    opening = confident & np.isin(classes, groups["openingCandidate"])
    excluded = binary_dilation(excluded, iterations=1)
    valid_alpha=np.asarray(original)[...,3]>0
    keep &= ~excluded & valid_alpha
    opening &= ~excluded & valid_alpha
    rgba = np.asarray(original).copy()
    rgba[..., 3][excluded] = 0
    eligible = np.asarray(original).copy()
    eligible[..., 3] = np.where(keep & (eligible[..., 3] > 0), eligible[..., 3], 0).astype(np.uint8)
    output_class = np.where(confident, classes + 1, 0).astype(np.uint16)
    return {"masked": Image.fromarray(rgba, mode="RGBA"), "eligible": Image.fromarray(eligible, mode="RGBA"), "class": Image.fromarray(output_class, mode="I;16"), "confidence": _probability_png(confidence), "opening": Image.fromarray(np.where(opening, 255, 0).astype(np.uint8), mode="L"), "groups": {key: int(np.count_nonzero(mask)) for key, mask in (("keep", keep), ("excluded", excluded), ("openingCandidate", opening), ("unknown", ~(keep | excluded | opening)))}, "eligibleCoverage": float(np.mean(keep & (np.asarray(original)[..., 3] > 0))), "confidenceMean": float(np.mean(confidence)), "inferenceScale": scale, "classCoverage": {str(class_id): float(np.mean((classes == class_id) & confident)) for class_id in range(resized.shape[0])}}


def _load_model(model_id: str, model_dir: Path | None, cache_dir: Path, device: torch.device) -> tuple[Any, Any, Path]:
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(model_id, cache_dir=str(cache_dir), allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors", "tokenizer*", "*.json"]))
    processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=False)
    # Recent Transformers/PyTorch use safe weights-only deserialization for this
    # public state dict.  No custom code is loaded from the repository.
    model = AutoModelForSemanticSegmentation.from_pretrained(model_dir, trust_remote_code=False, use_safetensors=False, weights_only=True)
    model.eval().to(device)
    return processor, model, model_dir


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    thumb_size = 256
    columns = min(4, len(paths))
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 24)), (35, 35, 35))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((thumb_size, thumb_size))
        x = (index % columns) * thumb_size + (thumb_size - image.width) // 2
        y = (index // columns) * (thumb_size + 24) + (thumb_size - image.height) // 2
        sheet.paste(image, (x, y), image.getchannel('A'))
        draw.text(((index % columns) * thumb_size + 4, (index // columns) * (thumb_size + 24) + thumb_size + 3), path.stem, fill="white")
    sheet.save(output)


def run(input_path: Path, output: Path, model_id: str = DEFAULT_MODEL_ID, model_dir: Path | None = None, cache_dir: Path = DEFAULT_CACHE_DIR, threshold: float = 0.70, device_name: str = "auto", limit: int | None = None) -> dict[str, Any]:
    provenance = json.loads(input_path.read_text())
    crops = list(provenance.get("crops", []))
    if limit is not None:
        crops = crops[:limit]
    output.mkdir(parents=True, exist_ok=True)
    device = _device(device_name)
    processor, model, resolved_model_dir = _load_model(model_id, model_dir, cache_dir, device)
    id2label = getattr(model.config, "id2label", {})
    groups = label_groups(id2label)
    model_files = sorted(path for path in resolved_model_dir.rglob("*") if path.is_file() and path.stat().st_size > 0)
    model_hash = hashlib.sha256("".join(f"{path.name}:{_sha256(path)}" for path in model_files).encode()).hexdigest()
    records = []
    masked_paths = []
    for crop in crops:
        crop_path = Path(crop["path"])
        item: dict[str, Any] = {"index": crop.get("index"), "sourceId": crop.get("sourceId"), "captureDate": crop.get("captureDate"), "sourceImagePath": crop.get("imagePath"), "cropPath": str(crop_path), "worldRegistrationValidated": False, "facadeMeasurementsGenerated": 0, "status": "rejected"}
        try:
            image = Image.open(crop_path).convert("RGBA")
            result = infer_mask(image, processor, model, groups, threshold, device)
            stem = f"crop-{int(crop.get('index', len(records))):03d}"
            masked_path = output / f"{stem}-masked.png"
            class_path = output / f"{stem}-classes.png"
            confidence_path = output / f"{stem}-confidence.png"
            opening_path = output / f"{stem}-opening-candidate.png"
            eligible_path = output / f"{stem}-facade-eligible.png"
            result["masked"].save(masked_path)
            result["eligible"].save(eligible_path)
            result["class"].save(class_path)
            result["confidence"].save(confidence_path)
            result["opening"].save(opening_path)
            masked_paths.append(masked_path)
            item.update({"status": "diagnostic", "outputMasked": str(masked_path), "outputFacadeEligible": str(eligible_path), "outputClasses": str(class_path), "outputConfidence": str(confidence_path), "outputOpeningCandidate": str(opening_path), "coverage": result["groups"], "eligibleCoverage": result["eligibleCoverage"], "confidenceMean": result["confidenceMean"], "inferenceScale": result["inferenceScale"], "classCoverage": result["classCoverage"], "outputSha256": {"masked": _sha256(masked_path), "facadeEligible": _sha256(eligible_path), "classes": _sha256(class_path), "confidence": _sha256(confidence_path), "openingCandidate": _sha256(opening_path)}})
        except Exception as exc:
            item["reason"] = f"crop failed: {type(exc).__name__}: {exc}"
        records.append(item)
    contact_sheet = output / "contact-sheet-masked.png"
    _contact_sheet(masked_paths, contact_sheet)
    report = {"schema": "facade-obstruction-mask-diagnostic/v1", "worldRegistrationValidated": False, "facadeMeasurementsGenerated": 0, "texturesPublished": False, "evidenceScope": "semantic obstruction hypotheses only; no occlusion truth, image-to-building identity, opening geometry, registration, or admission claim", "input": str(input_path.resolve()), "inputSha256": _sha256(input_path), "model": {"id": model_id, "resolvedDirectory": str(resolved_model_dir.resolve()), "hash": model_hash, "labels": {str(key): str(value) for key, value in id2label.items()}, "groups": groups, "confidenceThreshold": threshold, "inferenceCapPx": INFERENCE_CAP, "device": str(device)}, "alphaPolicy": "obstruction RGBA preserves input alpha for keep, unknown, and low-confidence pixels; high-confidence excluded pixels are alpha-zero after one-pixel conservative dilation", "facadeEligiblePolicy": "separate RGBA retains input alpha only for high-confidence configured keep labels with nonzero input alpha; unknown, low-confidence, excluded, and opening-candidate pixels are alpha-zero", "contactSheet": str(contact_sheet) if masked_paths else None, "crops": records, "summary": {"requested": len(crops), "diagnosticCompleted": sum(item["status"] == "diagnostic" for item in records), "failed": sum(item["status"] == "rejected" for item in records), "modelKeepLabels": groups["keep"], "modelExcludedLabels": groups["exclude"], "openingCandidateLabels": groups["openingCandidate"], "meanFacadeEligibleCoverage": float(np.mean([item["eligibleCoverage"] for item in records if item["status"] == "diagnostic"])) if any(item["status"] == "diagnostic" for item in records) else 0.0}}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run(args.input, args.output, args.model_id, args.model_dir, args.cache_dir, args.confidence_threshold, args.device, args.limit)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
