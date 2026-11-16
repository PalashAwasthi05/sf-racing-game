"""Regression checks for the rear corner light guides."""
import numpy as np

import roadster as R
from parts import tail_leds


def test_tail_guides_follow_rear_lens_and_wrap_both_corners():
    guides = tail_leds()
    assert len(guides) == 4
    for part, finish, mesh in guides:
        assert part == "taillamp"
        assert finish == "tail_led"
        # Every swept vertex stays behind the body skin and inside the lamp cutout.
        assert np.max(R.skin(*mesh.V.T)) < -0.002
        assert np.max(R.tail_outline(*mesh.V.T)) < 0

        centers = mesh.V[mesh.T].mean(axis=1)
        fn = mesh.face_normals()
        # Check the two visible runs independently. Face winding must agree with
        # the mesh's outward profile normals, including across the corner.
        positive = centers[:, 2] > 0
        rear = positive & (centers[:, 0] < -1.95) & (centers[:, 0] > -1.99)
        flank = positive & (centers[:, 0] > -1.75) & (centers[:, 0] < -1.65)
        assert rear.sum() > 20 and flank.sum() > 20
        outward = mesh.N[mesh.T].mean(axis=1)
        outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-12)
        agreement = np.sum(fn * outward, axis=1)
        assert np.median(agreement[rear]) > 0.9
        assert np.median(agreement[flank]) > 0.9

