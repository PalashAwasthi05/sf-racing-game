"""Regression for host-mask subtraction leaving an unskinned wall patch.

v0.12 recorded this defect as a toy (a padded frontage mask minus a smaller
emitted skin left 56 m2 of nothing). v0.14 closes it in the generator: the mask
is the union of emitted root-skin slots, so the same construction has no gap.
"""
import numpy as np
from shapely.geometry import box
from pipeline import envelopes

PROFILE = {'id': 'gap-test', 'match': {'featureId': 'test'}, 'normal': [0, 1], 'replaceDepthM': 3, 'levels': {},
           'nodes': [{'id': 'only-skin', 'span': [0.2, 0.8], 'z': [0, 8], 'material': 'stone', 'depthM': 0}]}


def missing_host_patch():
    host_frontage = box(0.0, -3.0, 10.0, 0.0)
    c = {'id': 'test', 'g': box(0, -10, 10, 0)}
    resolved = envelopes.prepare({}, c, 0, np.empty((0, 3)), [PROFILE])
    cut = envelopes.replacement_mask(resolved).intersection(host_frontage)
    emitted_plan = box(2.0, -3.0, 8.0, 0.0)          # the one skin covers 20%..80% of the frontage
    return cut.difference(emitted_plan)


def test_replacement_mask_must_be_covered_by_emitted_skin():
    gap = missing_host_patch()
    assert gap.area < 1e-6, gap.area


if __name__ == "__main__":
    gap = missing_host_patch()
    print({"missingAreaM2": gap.area})
