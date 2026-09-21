"""Evidence driven loader for generic ambientCG material inventory."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from PIL import Image
from trimesh.visual.material import PBRMaterial

ROOT = Path(__file__).resolve().parents[1] / "material-library-v9"

@dataclass(frozen=True)
class MaterialAsset:
    record: dict[str, Any]
    material: PBRMaterial | None
    uv_scale_m: tuple[float, float] | None
    provenance: dict[str, Any]
    resolved: bool
    reason: str | None = None

_CACHE: dict[str, MaterialAsset] = {}

def inventory(manifest_path: str | Path = ROOT / "manifest.json") -> list[dict[str, Any]]:
    """Return generic category records; no scene or geography is consulted."""
    data = json.loads(Path(manifest_path).read_text())
    return list(data.get("assets", []))

def _find(asset_root: Path, token: str) -> Path | None:
    hits = sorted(p for p in asset_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"} and token in p.stem.lower())
    return hits[0] if hits else None

def _unresolved(record, reason, override=None):
    scale = tuple(override) if override is not None else None
    return MaterialAsset(dict(record), None, scale, {"source_url": record.get("source_url"), "asset_id": record.get("asset_id")}, False, reason)

def load_asset(record: Mapping[str, Any], root: str | Path = ROOT, measured_scale_m=None) -> MaterialAsset:
    """Load one manifest record; measured scale may resolve source unknowns."""
    rec = dict(record); aid = rec.get("asset_id"); category = rec.get("category")
    if not aid or not category: return _unresolved(rec, "missing asset identity")
    scale = measured_scale_m if measured_scale_m is not None else rec.get("tile_size_m")
    if scale is None: return _unresolved(rec, "tile_size_m unknown; measured override required", measured_scale_m)
    try: scale = tuple(float(x) for x in scale)
    except (TypeError, ValueError): return _unresolved(rec, "invalid tile_size_m", measured_scale_m)
    if len(scale) != 2 or not np.all(np.isfinite(scale)) or min(scale) <= 0: return _unresolved(rec, "invalid tile_size_m", measured_scale_m)
    root = Path(root).resolve(); base = (root / str(category) / str(aid)).resolve()
    try: base.relative_to(root)
    except ValueError: return _unresolved(rec, "asset path escapes supplied root", measured_scale_m)
    color, normal, rough = _find(base, "color"), _find(base, "normalgl"), _find(base, "roughness")
    if not color or not rough: return _unresolved(rec, "missing color or roughness map", measured_scale_m)
    if not normal: return _unresolved(rec, "missing OpenGL normal map", measured_scale_m)
    try:
        c, n, r = Image.open(color).convert("RGB"), Image.open(normal).convert("RGB"), Image.open(rough).convert("L")
        sizes = {c.size, n.size, r.size}
        if len(sizes) != 1 or any(max(width, height) < 1024 for width, height in sizes): return _unresolved(rec, "maps are below 1K or dimensions differ", measured_scale_m)
    except Exception as exc: return _unresolved(rec, f"invalid image: {exc}", measured_scale_m)
    metal_path = _find(base, "metalness") or _find(base, "metallic")
    metal = Image.open(metal_path).convert("L") if metal_path else Image.new("L", r.size, 0)
    if metal.size != r.size: return _unresolved(rec, "metalness dimensions differ", measured_scale_m)
    fingerprint = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (color, normal, rough, metal_path) if p)
    key = (rec.get("archive_sha256") or "", str(base.resolve()), scale, fingerprint)
    if key in _CACHE: return _CACHE[key]
    mr = Image.merge("RGB", (Image.new("L", r.size, 0), r, metal))
    mat = PBRMaterial(name=str(aid), baseColorTexture=c, normalTexture=n, metallicRoughnessTexture=mr, metallicFactor=1.0 if metal_path else 0.0, roughnessFactor=1.0, baseColorFactor=[255,255,255,255])
    result = MaterialAsset(rec, mat, scale, {"source_url": rec.get("source_url"), "download_url": rec.get("download_url"), "license": rec.get("license"), "archive_sha256": rec.get("archive_sha256"), "maps": {"color": str(color), "normal_gl": str(normal), "roughness": str(rough), "metalness": str(metal_path) if metal_path else None}}, True)
    _CACHE[key] = result; return result

def candidates(category: str, records=None, observed: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Return category candidates; descriptors only rank evidence, never assign."""
    rows = [dict(x) for x in (records if records is not None else inventory()) if x.get("category") == category]
    if not observed: return rows
    def score(row):
        d = row.get("descriptors") or {}
        missing = sum(k not in d for k in observed)
        distance = sum((float(d[k])-float(v))**2 for k,v in observed.items() if k in d)
        return (missing, distance)
    return sorted(rows, key=score)
