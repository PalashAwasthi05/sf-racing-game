import numpy as np
import torch
from PIL import Image

from mask_facade_obstructions_v12 import infer_mask, label_groups


class FakeProcessor:
    def __call__(self, **kwargs):
        return {"pixel_values": torch.zeros((1, 3, 512, 512), dtype=torch.float32)}


class FakeModel:
    def __call__(self, pixel_values):
        height, width = pixel_values.shape[-2] // 4, pixel_values.shape[-1] // 4
        logits = torch.zeros((1, 4, height, width), dtype=torch.float32)
        logits[:, 1, : height * 2 // 3, : width // 3] = 8.0  # configured building
        logits[:, 2, : height * 2 // 3, width // 3 : 2 * width // 3] = 8.0  # configured sky
        logits[:, 3, : height * 2 // 3, 2 * width // 3 :] = 8.0  # configured car
        return type("Output", (), {"logits": logits})()


def test_label_groups_are_configuration_driven():
    groups = label_groups({"0": "unknown", "1": "building", "2": "sky", "3": "car", "4": "windowpane"})
    assert groups["keep"] == [1]
    assert groups["exclude"] == [2, 3]
    assert groups["openingCandidate"] == [4]

def test_label_substrings_do_not_invent_building_or_car_classes():
    groups=label_groups({0:'wall clock',1:'carpet',2:'house',3:'tree',4:'screen door'})
    assert groups['unknown']==[0,1]
    assert groups['keep']==[2] and groups['exclude']==[3]
    assert groups['openingCandidate']==[4]


def test_excluded_classes_mask_and_low_confidence_preserves_alpha():
    alpha = np.full((512, 1024), 127, dtype=np.uint8)
    rgba = np.dstack((np.full((512, 1024, 3), 80, dtype=np.uint8), alpha))
    result = infer_mask(Image.fromarray(rgba, mode="RGBA"), FakeProcessor(), FakeModel(), {"keep": [1], "exclude": [2, 3], "openingCandidate": [], "unknown": [0]}, 0.70, torch.device("cpu"))
    output = np.asarray(result["masked"])
    eligible = np.asarray(result["eligible"])
    assert output[100, 100, 3] == 127  # building/kept region
    assert output[100, 900, 3] == 0  # high-confidence excluded region
    assert output[450, 100, 3] == 127  # no confident class: unknown preserves input alpha
    assert eligible[100, 100, 3] == 127  # only confident building/wall is eligible
    assert eligible[100, 900, 3] == 0
    assert eligible[450, 100, 3] == 0  # unknown is never texture-eligible
    assert np.all(eligible[...,3] <= output[...,3])  # dilated obstruction boundary wins
