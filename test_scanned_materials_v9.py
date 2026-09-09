from pathlib import Path
import json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from general_world.scanned_materials import inventory, load_asset, candidates, _CACHE

def _fixture(tmp_path, *, normal=True, scale=(2.0, 1.0), metal=False):
    root = tmp_path / "library"; base = root / "brick" / "sample" / "maps"; base.mkdir(parents=True)
    Image.new("RGB", (1024, 1024), (128, 90, 60)).save(base / "Sample_1K_Color.jpg")
    if normal: Image.new("RGB", (1024, 1024), (128, 128, 255)).save(base / "Sample_1K_NormalGL.jpg")
    Image.new("L", (1024, 1024), 180).save(base / "Sample_1K_Roughness.jpg")
    if metal: Image.new("L", (1024, 1024), 255).save(base / "Sample_1K_Metalness.jpg")
    return root, {"asset_id":"sample", "category":"brick", "tile_size_m":list(scale), "source_url":"https://example.invalid/a/sample", "download_url":"https://example.invalid/get/sample", "license":"CC0-1.0", "archive_sha256":"fixture"}

def test_load_valid_asset_and_roughness_channel(tmp_path):
    root, rec = _fixture(tmp_path)
    a = load_asset(rec, root)
    assert a.resolved and a.uv_scale_m == (2.0, 1.0)
    tex = np.asarray(a.material.metallicRoughnessTexture)
    assert tex.shape[-1] == 3 and int(tex[20,20,1]) == 180 and int(tex[20,20,2]) == 0

def test_unknown_scale_requires_measured_override(tmp_path):
    root, rec = _fixture(tmp_path); rec["tile_size_m"] = None
    assert not load_asset(rec, root).resolved
    assert load_asset(rec, root, measured_scale_m=(.5,.5)).resolved

def test_missing_normal_is_unresolved(tmp_path):
    root, rec = _fixture(tmp_path, normal=False)
    result = load_asset(rec, root)
    assert not result.resolved and "normal" in result.reason

def test_metalness_is_used_only_when_source_map_exists(tmp_path):
    root, rec = _fixture(tmp_path, metal=True)
    a = load_asset(rec, root); assert a.resolved and a.material.metallicFactor == 1.0
    root2, rec2 = _fixture(tmp_path / "dielectric")
    b = load_asset(rec2, root2); assert b.resolved and b.material.metallicFactor == 0.0

def test_category_candidates_do_not_assign_or_use_building_rules():
    rows = [{"category":"brick", "asset_id":"a", "descriptors":{"roughness":.8}}, {"category":"brick", "asset_id":"b", "descriptors":{"roughness":.2}}, {"category":"metal", "asset_id":"c"}]
    assert [x["asset_id"] for x in candidates("brick", rows, {"roughness":.2})] == ["b", "a"]
    assert candidates("masonry", rows) == []

def test_shared_cache_reuses_loaded_material(tmp_path):
    _CACHE.clear(); root, rec = _fixture(tmp_path)
    assert load_asset(rec, root).material is load_asset(rec, root).material

def test_inventory_reads_manifest_only(tmp_path):
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"assets":[{"category":"stone","asset_id":"x"}]}))
    assert inventory(manifest) == [{"category":"stone","asset_id":"x"}]
