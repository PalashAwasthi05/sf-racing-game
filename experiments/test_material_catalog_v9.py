"""Contract tests for metric, structural material recipes."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from material_catalog_v9 import MaterialCatalog

def test_metric_scale_changes_joint_layout_and_is_deterministic():
    c = MaterialCatalog(96)
    a = c.resolve("masonry", {"kind":"brick", "bond":"running", "scale_m":(.86,.276)})
    b = c.resolve("masonry", {"kind":"brick", "bond":"running", "scale_m":(1.72,.552)})
    again = c.resolve("masonry", {"kind":"brick", "bond":"running", "scale_m":(.86,.276)})
    assert a.scale_m == (.86,.276) and a.recipe_hash == again.recipe_hash
    # Metric sampling changes the number and position of mortar bands in pixels.
    assert not np.array_equal(np.asarray(a.maps["height"]) if "height" in a.maps else np.asarray(a.maps["normal"]), np.asarray(b.maps["normal"]))

def test_bonds_have_distinct_height_and_normal_structure():
    c = MaterialCatalog(96)
    recipes = [c.resolve("masonry", {"kind":"brick", "bond":bond}) for bond in ("running","stack","English","Flemish")]
    normals = [np.asarray(r.maps["normal"]) for r in recipes]
    assert all(r.resolved for r in recipes)
    assert len({n.tobytes() for n in normals}) == 4
    assert any(np.std(n[...,0]) > 0 for n in normals)

def test_profiles_and_unknown_structural_variants():
    c = MaterialCatalog(64)
    corrugated = c.resolve("metal", {"panel":"corrugated"})
    standing = c.resolve("metal", {"panel":"standing_seam"})
    boards = c.resolve("timber", {"product":"boards"})
    shingles = c.resolve("timber", {"product":"shingles"})
    assert not np.array_equal(np.asarray(corrugated.maps["normal"]), np.asarray(standing.maps["normal"]))
    assert not np.array_equal(np.asarray(boards.maps["normal"]), np.asarray(shingles.maps["normal"]))
    assert c.resolve("masonry", {"kind":"brick", "bond":"diagonal"}).resolved is False
    assert c.resolve("masonry", {"kind":"brick", "bond":"ashlar"}).resolved is False

def test_metal_dielectric_and_measured_srgb_color():
    c = MaterialCatalog(64)
    painted = c.resolve("metal", {"finish":"painted", "color_srgb":(128,64,32)})
    polished = c.resolve("metal", {"finish":"polished"})
    assert painted.metallic == 0.0 and polished.metallic == 1.0
    assert tuple(np.asarray(painted.maps["color"])[32,32]) != tuple(np.asarray(polished.maps["color"])[32,32])

def test_unknown_family_is_unresolved_and_flat_glass_is_allowed():
    c = MaterialCatalog(64)
    unknown = c.resolve("building_type_17")
    glass = c.resolve("glass", {"kind":"clear", "finish":"low_iron"})
    assert not unknown.resolved and unknown.maps == {}
    # Smooth glass has a physically acceptable flat normal field.
    assert np.unique(np.asarray(glass.maps["normal"]), axis=0).shape[0] == 1
