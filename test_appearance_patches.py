import numpy as np
from PIL import Image

from regen.appearance_patches import crop_zone, detector_valid, exclusion_mask, host_surface_mask, normalize_median, residuals


def test_crop_zone_uses_metric_coordinates_and_removes_padding():
    # 4x3 content at 2 px/m, surrounded by 1px annotation border.
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[1:4, 1:5] = 100
    image = Image.fromarray(rgb)
    view = {"rectPad": 1, "ppm": 2.0}
    zone = {"x0": 0.5, "x1": 1.5, "z0": 0.5, "z1": 1.5}
    crop, box = crop_zone(image, view, wall_height=2.0, zone=zone)
    assert crop.shape == (2, 2, 3)
    assert box.tolist() == [1, 1, 3, 3]
    assert np.all(crop == 100)


def test_opening_and_semantic_masks_exclude_pixels_without_changing_other_area():
    zone = {"x0": 0, "x1": 4, "z0": 0, "z1": 4}
    reading = {"openings": [{"x0": 1, "x1": 2, "z0": 1, "z1": 2}], "banks": []}
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0, 0] = 3
    valid = exclusion_mask((4, 4), np.array([0, 0, 4, 4]), 1, 4, zone, reading, labels)
    assert not valid[0, 0]  # semantic occluder
    assert not valid[2, 1]  # modeled opening; image y is wall z-up reversed
    assert valid.sum() == 14


def test_patch_normalization_preserves_texture_and_reports_flat_baseline():
    yy, xx = np.mgrid[:32, :32]
    source = np.stack([100 + 10 * (xx % 2), 90 + 8 * (yy % 2), np.full_like(xx, 80)], axis=-1).astype(np.uint8)
    valid = np.ones((32, 32), dtype=bool)
    patch, gain = normalize_median(source, valid, "#b4a28c")
    assert patch.shape == source.shape
    assert np.allclose(np.median(patch, axis=(0, 1)), [180, 162, 140], atol=1)
    assert len(gain) == 3
    metrics, _, _ = residuals(source, source.copy(), valid, "#b4a28c")
    assert metrics["validPixels"] == 1024
    assert metrics["swatchNormalizedPatchVsHeldoutMedianDeltaE76"] == 0
    assert metrics["swatchNormalizedTextureHighpassCorrelation"] > .99
    assert metrics["rawTextureHighpassCorrelation"] > .99
    assert metrics["rawSourceTextureHighpassStdSrgbLuma"] > 0


def test_detector_boxes_are_excluded_from_texture_mask():
    valid = detector_valid((6, 8), [(2, 1, 4, 3, "window", .8)])
    assert valid.sum() == 8 * 6 - 9
    assert not valid[2, 3]
    assert valid[0, 0]


def test_host_surface_mask_excludes_projecting_bay_but_keeps_window_features():
    zone = {"x0": 0, "x1": 4, "z0": 0, "z1": 4}
    reading = {"bays": [{"x0": 2, "x1": 4, "z0": 1, "z1": 3}]}
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0, 0] = 1  # opening/window class remains available as a host-wall alignment landmark
    valid = host_surface_mask((4, 4), np.array([0, 0, 4, 4]), 1, 4, zone, reading, labels)
    assert valid[0, 0]
    assert not valid[2, 2]
    assert valid[0, 3]
