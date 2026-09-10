import numpy as np

import torch

from propose_facade_parts_v12 import _cache_key, alpha_overlap, canonical_label, clip_box, filter_detections


def test_clip_box_rejects_nonfinite_and_clips_bounds():
    assert clip_box([-4, 2, 12, 9], 10, 8) == (0.0, 2.0, 10.0, 8.0)
    assert clip_box([np.nan, 0, 1, 1], 10, 8) is None
    assert clip_box([3, 3, 3, 4], 10, 8) is None


def test_alpha_overlap_and_padding_rejection():
    alpha = np.zeros((10, 10), dtype=np.uint8)
    alpha[2:8, 2:8] = 255
    assert alpha_overlap(alpha, (2, 2, 8, 8)) == 1.0
    assert alpha_overlap(alpha, (0, 0, 2, 2)) == 0.0
    proposals = filter_detections([[-2, -2, 2, 2], [2, 2, 8, 8]], [0.9, 0.8], ["window", "door"], alpha)
    assert len(proposals) == 1 and proposals[0]["category"] == "door"


def test_labels_remain_generic_categories():
    assert canonical_label("a large window") == "window"
    assert canonical_label("balconies") == "balcony"
    assert canonical_label("tree") == "unknown"


def test_cache_key_changes_when_crop_or_threshold_changes():
    device = torch.device("cpu")
    base = _cache_key("crop-a", "model-a", box_threshold=0.25, text_threshold=0.2, alpha_minimum=0.25, device=device)
    assert base != _cache_key("crop-b", "model-a", box_threshold=0.25, text_threshold=0.2, alpha_minimum=0.25, device=device)
    assert base != _cache_key("crop-a", "model-a", box_threshold=0.30, text_threshold=0.2, alpha_minimum=0.25, device=device)
