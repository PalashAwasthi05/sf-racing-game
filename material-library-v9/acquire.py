#!/usr/bin/env python3
"""Acquire generic ambientCG 1K PBR sets with source metadata and checksums.

The only input is a material category; no scene or location data is accepted.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile, urllib.parse, urllib.request, zipfile
from pathlib import Path
from PIL import Image

API = "https://ambientcg.com/api/v2/full_json"
CATEGORIES = {
    "brick": ["Bricks076A", "Bricks075B", "Bricks088"],
    "stone": ["Rocks022", "Travertine013", "PavingStones139"],
    "plaster": ["Plaster001", "Plaster007", "PaintedPlaster017"],
    "concrete": ["Concrete039", "Concrete034", "Concrete046"],
    "metal": ["MetalPlates014", "MetalPlates012", "CorrugatedSteel009"],
    "timber": ["WoodFloor034", "Wood090A", "Planks021"],
    "roofing": ["RoofingTiles006", "RoofingTiles008", "Tiles098"],
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def metadata():
    # The API's first 100 records include all selected popular records; follow
    # pagination so this remains reusable if the ranking changes.
    found = {}
    for offset in range(0, 3000, 100):
        query = urllib.parse.urlencode({"limit": 100, "offset": offset, "include": "technicalData"})
        with urllib.request.urlopen(API + "?" + query, timeout=60) as r: page = json.load(r)
        for item in page.get("foundAssets", []): found[item["assetId"]] = item
        if len(page.get("foundAssets", [])) < 100: break
    return found

def acquire(categories, out):
    if len(categories) * 3 > 24: raise ValueError("maximum 24 assets")
    records = metadata(); out.mkdir(parents=True, exist_ok=True); manifest = []
    for category in categories:
        for asset_id in CATEGORIES[category]:
            item = records.get(asset_id)
            if not item or item.get("dataTypeName") != "Material": raise RuntimeError(f"missing material metadata: {asset_id}")
            maps = set(item.get("maps", []))
            if not {"color", "normal", "roughness"} <= maps: raise RuntimeError(f"incomplete PBR maps: {asset_id}")
            tile = None if not item.get("dimensionX") or not item.get("dimensionY") else [item["dimensionX"] / 100, item["dimensionY"] / 100]
            asset_dir = out / category / asset_id; asset_dir.mkdir(parents=True, exist_ok=True)
            url = f"https://ambientcg.com/get?file={asset_id}_1K-JPG.zip"
            archive = asset_dir / (asset_id + "_1K-JPG.zip")
            if not archive.exists():
                with urllib.request.urlopen(url, timeout=120) as src, open(archive, "wb") as dst: shutil.copyfileobj(src, dst)
            if archive.stat().st_size > 12 * 1024 * 1024: raise RuntimeError(f"unexpectedly large archive: {asset_id}")
            if sum(x.get("archive_bytes", 0) for x in manifest) + archive.stat().st_size > 200 * 1024 * 1024: raise RuntimeError("download budget exceeded")
            with zipfile.ZipFile(archive) as z:
                names = z.namelist(); lower = " ".join(names).lower()
                if not all(token in lower for token in ("color", "normalgl", "roughness")): raise RuntimeError(f"archive missing 1K OpenGL map: {asset_id}")
                z.extractall(asset_dir / "maps")
            check = list((asset_dir / "maps").rglob("*.jpg"))
            if not any("normalgl" in x.name.lower() and Image.open(x).width >= 1024 for x in check): raise RuntimeError(f"normalGL is not 1K: {asset_id}")
            manifest.append({"asset_id": asset_id, "category": category, "source_url": f"https://ambientcg.com/a/{asset_id}", "download_url": url, "license": "CC0-1.0", "license_url": "https://docs.ambientcg.com/license/", "creation_method": item.get("creationMethodName"), "tile_size_m": tile, "tile_size_source": "ambientCG dimensionX/dimensionY centimeters" if tile else "unknown in primary metadata", "maps": sorted(maps), "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size})
    (out / "manifest.json").write_text(json.dumps({"provider":"ambientCG","license":"CC0-1.0","asset_count":len(manifest),"assets":manifest}, indent=2) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--category", action="append", choices=sorted(CATEGORIES), required=True); ap.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = ap.parse_args(); acquire(args.category, args.output)
