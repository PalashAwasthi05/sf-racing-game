import numpy as np
import cv2

from regen.panorama_gravity import _fit_gravity, corrected_rotation, estimate_panorama_gravity


def _noisy_vertical_line_plane_normals(gravity, count=40, seed=7):
    rng = np.random.default_rng(seed)
    normals, sectors = [], []
    for index in range(count):
        seed_axis = np.eye(3)[index % 3]
        normal = np.cross(gravity, seed_axis)
        if np.linalg.norm(normal) < .1:
            normal = np.cross(gravity, np.array([1., 0., 0.]))
        normal = normal / np.linalg.norm(normal)
        normal += rng.normal(0., .002, 3)
        normals.append(normal / np.linalg.norm(normal))
        sectors.append(index % 6)
    return np.asarray(normals), np.asarray(sectors)


def test_consensus_recovers_tilted_camera_up_amid_line_outliers():
    expected = np.array([-.3896, -.9204, -.0328])
    expected /= np.linalg.norm(expected)
    normals, sectors = _noisy_vertical_line_plane_normals(expected)
    rng = np.random.default_rng(12)
    outliers = rng.normal(size=(28, 3))
    outliers /= np.linalg.norm(outliers, axis=1, keepdims=True)
    all_normals = np.vstack([normals, outliers])
    all_sectors = np.r_[sectors, rng.integers(0, 8, len(outliers))]

    evidence = _fit_gravity(all_normals, np.ones(len(all_normals)), all_sectors,
                            min_support=20, min_sectors=3)
    assert evidence['accepted']
    assert evidence['support'] >= 38
    assert evidence['support_sectors'] == 6
    assert np.degrees(np.arccos(np.clip(np.asarray(evidence['gravity']) @ expected, -1., 1.))) < .5
    assert evidence['median_residual_deg'] < .2


def test_consensus_rejects_a_single_azimuth_facet_even_with_many_lines():
    expected = np.array([-.25, -.96, .12])
    expected /= np.linalg.norm(expected)
    normals, _ = _noisy_vertical_line_plane_normals(expected, count=30)
    evidence = _fit_gravity(normals, np.ones(len(normals)), np.zeros(len(normals), dtype=int),
                            min_support=20, min_sectors=3)
    assert not evidence['accepted']
    assert evidence['reason'] == 'no gravity consensus spans enough azimuth sectors'


def test_isotropic_random_line_planes_fail_fraction_and_consensus_quality_checks():
    rng = np.random.default_rng(29)
    normals = rng.normal(size=(500, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    evidence = _fit_gravity(normals, np.ones(len(normals)), rng.integers(0, 8, len(normals)),
                            min_support=20, min_sectors=3)
    assert not evidence['accepted']
    assert evidence['reason'] in (
        'consensus fraction is too small for non-random line evidence',
        'distinct gravity hypotheses have similar support',
        'line-plane normals do not define a unique gravity direction',
    )


def test_parallel_line_plane_normals_are_rejected_as_gravity_degenerate():
    normals = np.tile(np.array([1., 0., 0.]), (40, 1))
    sectors = np.arange(40) % 6
    evidence = _fit_gravity(normals, np.ones(len(normals)), sectors,
                            prior=[0., -1., 0.], min_support=20, min_sectors=3)
    assert not evidence['accepted']
    assert evidence['eigen_gap'] < .04
    assert evidence['reason'] == 'line-plane normals do not define a unique gravity direction'


def test_distributed_true_support_outvotes_more_sector_coverage_in_low_support_noise():
    expected = np.array([-.3896, -.9204, -.0328])
    expected /= np.linalg.norm(expected)
    normals, sectors = _noisy_vertical_line_plane_normals(expected, count=300, seed=17)
    sectors %= 7
    rng = np.random.default_rng(18)
    noise = rng.normal(size=(20, 3))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    all_normals = np.vstack([normals, noise])
    all_sectors = np.r_[sectors, rng.integers(0, 8, len(noise))]
    evidence = _fit_gravity(all_normals, np.ones(len(all_normals)), all_sectors,
                            min_support=20, min_sectors=3)
    assert evidence['accepted']
    assert evidence['support'] >= 295
    assert np.degrees(np.arccos(np.clip(np.asarray(evidence['gravity']) @ expected, -1., 1.))) < .5


def test_corrected_rotation_maps_metadata_up_to_pixel_estimate_without_reflection():
    angle = np.radians(23.)
    around_x = np.array([[1., 0., 0.],
                         [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
    around_z = np.array([[np.cos(.4), -np.sin(.4), 0.],
                         [np.sin(.4), np.cos(.4), 0.],
                         [0., 0., 1.]])
    rotation = around_z @ around_x
    observed_up = np.array([-.3896, -.9204, -.0328])
    observed_up /= np.linalg.norm(observed_up)

    corrected = corrected_rotation(rotation, observed_up)
    assert np.allclose(corrected @ [0., 0., 1.], observed_up, atol=1e-10)
    assert np.isclose(np.linalg.det(corrected), 1., atol=1e-10)
    assert np.allclose(corrected.T @ corrected, np.eye(3), atol=1e-10)


def test_panorama_pixels_recover_tilted_up_with_nonarchitectural_line_outliers():
    gravity = np.array([-.3896, -.9204, -.0328])
    gravity /= np.linalg.norm(gravity)
    image = np.zeros((512, 1024), dtype=np.uint8)
    e1 = np.cross(gravity, [0., 0., 1.]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(gravity, e1)

    def draw_line(points, color=220):
        rays = points / np.linalg.norm(points, axis=1, keepdims=True)
        yaw = np.arctan2(rays[:, 0], rays[:, 2])
        pitch = np.arcsin(np.clip(-rays[:, 1], -1., 1.))
        xy = np.c_[(yaw / (2 * np.pi) + .5) * image.shape[1],
                   (.5 - pitch / np.pi) * image.shape[0]].astype(np.int32)
        # Split at the equirectangular seam instead of drawing a spurious line across the pano.
        starts = np.r_[0, np.flatnonzero(np.abs(np.diff(xy[:, 0])) > image.shape[1] // 2) + 1]
        ends = np.r_[starts[1:], len(xy)]
        for start, end in zip(starts, ends):
            if end - start >= 2:
                cv2.polylines(image, [xy[start:end].reshape(-1, 1, 2)], False, color, 2)

    for i, angle in enumerate(np.linspace(0., 2 * np.pi, 32, endpoint=False)):
        direction = np.cos(angle) * e1 + np.sin(angle) * e2
        distance = 12. + (i % 3)
        t = np.linspace(-4., 4., 90)
        draw_line(distance * direction + t[:, None] * gravity)

    rng = np.random.default_rng(14)
    for _ in range(16):
        center = rng.normal(size=3); center[2] += 10.
        direction = rng.normal(size=3)
        t = np.linspace(-3., 3., 40)
        draw_line(center + t[:, None] * direction, color=140)

    evidence = estimate_panorama_gravity(image, view_size=300, min_support=20, min_sectors=3)
    assert evidence['accepted']
    assert evidence['support_sectors'] >= 3
    assert np.degrees(np.arccos(np.clip(np.asarray(evidence['gravity']) @ gravity, -1., 1.))) < 1.5
