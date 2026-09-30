"""See-through fences found in the LiDAR (frontage_fences_v36) mask the wall behind them."""
import numpy as np
import frontage_evidence_v34 as E


def test_an_opening_standing_behind_a_fence_is_dropped_and_one_above_it_kept():
    H, L = 4., 12.; mask = np.zeros((int(H * E.PPM), int(L * E.PPM)), bool)
    mask[int((H - 2.1) * E.PPM):, :] = True                        # a 2.1 m fence hides the wall's foot
    shop = (0., .1, 12., 3.3); high = (2., 2.4, 4., 3.8)
    assert E.box_hidden(shop, mask, H) >= .3 and E.box_hidden(high, mask, H) < .3
