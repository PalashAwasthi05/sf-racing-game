import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "regen"))
from measure_surface_modules import compare_views, measure_view


def _grid(row_pitch=24, col_pitch=40, *, h=320, w=400):
    y, x = np.mgrid[:h, :w]
    gray = np.full((h, w), 180., dtype=np.float32)
    if row_pitch:
        gray[(y % row_pitch) < 2] = 80
    if col_pitch:
        gray[(x % col_pitch) < 2] = 120
    return np.repeat(gray[..., None], 3, axis=2)


def test_masked_repeat_requires_independent_strips_and_converts_pixels_to_metres():
    image = _grid()
    mask = np.ones(image.shape[:2], dtype=bool)
    # Openings/occluders remove contiguous areas, but the remaining wall strips
    # still independently carry both repeat axes.
    mask[50:110, 100:150] = False
    mask[210:260, 270:320] = False
    a = measure_view(image, mask, 45.)
    b = measure_view(image, mask, 45.)
    assert a["accepted"] and b["accepted"]
    assert abs(a["axes"]["vertical_repeat_px"]["period_px"] - 24) < .1
    assert abs(a["axes"]["horizontal_repeat_px"]["period_px"] - 40) < .1
    assert abs(a["axes"]["vertical_repeat_px"]["period_m"] - 24 / 45) < .002
    assert compare_views(a, b)["accepted"]


def test_two_views_with_different_periods_do_not_produce_a_module_proposal():
    mask = np.ones((320, 400), dtype=bool)
    a = measure_view(_grid(row_pitch=24, col_pitch=None), mask, 45.)
    b = measure_view(_grid(row_pitch=32, col_pitch=None), mask, 45.)
    agreed = compare_views(a, b)
    assert not agreed["axes"]["vertical_repeat_px"]["accepted"]
    assert not agreed["accepted"]


def test_below_nyquist_texture_is_reported_as_unresolvable():
    _, x = np.mgrid[:320, :400]
    gray = np.repeat(((x % 2) == 0)[..., None], 3, axis=2).astype(np.uint8) * 140
    result = measure_view(gray, np.ones(gray.shape[:2], dtype=bool), 45.)
    repeats = result["axes"]["horizontal_repeat_px"]["strips"]
    assert all(not r["accepted"] for r in repeats)
    assert all(r.get("belowNyquist") for r in repeats)


def test_low_valid_wall_coverage_abstains_before_period_search():
    image = _grid()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:79] = True
    result = measure_view(image, mask, 45.)
    assert not result["accepted"]
    assert result["reason"] == "valid wall coverage below acceptance gate"


def test_unstructured_noise_does_not_create_repeat_from_peak_search():
    image = np.random.default_rng(42).integers(70, 210, (320, 400, 3), dtype=np.uint8)
    result = measure_view(image, np.ones(image.shape[:2], dtype=bool), 45.)
    assert not result["accepted"]

