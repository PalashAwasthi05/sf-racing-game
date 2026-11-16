import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "regen"))
from project_surface_photos import (build_detail_atlas, select_host_faces,
                                    wall_patch_coordinates, _split_mesh,
                                    resample_map_to_source_frame,
                                    _prefilter_for_uv_footprint)
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial


def _triangle(points):
    return trimesh.Trimesh(vertices=np.asarray(points, dtype=float), faces=[[0, 1, 2]], process=False)


def test_projection_selects_only_outward_triangles_on_observed_plane_and_zone():
    # Plane x=0, outward normal +x; profile tangent points along +y.
    wall = [[0, 0, 0], [0, 2, 0], [0, 0, 2]]
    wrong_plane = [[.2, 0, 0], [.2, 2, 0], [.2, 0, 2]]
    reverse = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]
    outside_zone = [[0, 3, 0], [0, 5, 0], [0, 3, 2]]
    verts = np.asarray(wall + wrong_plane + reverse + outside_zone, dtype=float)
    faces = np.arange(12).reshape(4, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    selected, counts = select_host_faces(mesh, normal=[1, 0], plane_v=0,
                                         frame_u=[0, 5], ground_z=0,
                                         zone={"x0": 0, "x1": 2, "z0": 0, "z1": 2})
    assert selected.tolist() == [True, False, False, False]
    assert counts["selectedFaces"] == 1


def test_wall_frame_mapping_keeps_source_pixel_axes_and_vertical_origin():
    vertices = np.asarray([[0, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    pixels = wall_patch_coordinates(vertices, [1, 0], [0, 2], 0,
                                    {"x0": 0, "x1": 2, "z0": 0, "z1": 2}, 10)
    assert np.allclose(pixels, [[0, 20], [20, 20], [0, 0]])


def test_photo_detail_only_changes_valid_mask_and_preserves_base_hue():
    h, w = 40, 48
    yy, xx = np.mgrid[:h, :w]
    checker = (((xx // 2 + yy // 2) % 2) * 100 + 70).astype(np.uint8)
    photo = np.repeat(checker[..., None], 3, axis=2)
    mask = np.zeros((h, w), dtype=bool)
    mask[8:32, 9:39] = True
    base = Image.new("RGB", (8, 8), (105, 130, 150))
    uv = np.column_stack([np.zeros(h * w), np.zeros(h * w)])
    after, stats = build_detail_atlas(photo, mask, base, uv, strength=.4)
    before = np.asarray(base)[0, 0]
    out = np.asarray(after)
    assert np.array_equal(out[~mask], np.broadcast_to(before, out[~mask].shape))
    assert np.any(out[mask] != np.broadcast_to(before, out[mask].shape))
    # The photo contributes scalar luminance only; it does not replace the swatch hue.
    assert np.all(np.abs(out[mask].mean(axis=0) - before) < 10)
    assert stats["validFraction"] == mask.mean()


def test_mesh_split_keeps_invalid_triangles_and_positions_on_original_material():
    vertices = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 1],
                           [0, 1, 1], [0, 0, 2], [0, 1, 2]], dtype=float)
    faces = np.asarray([[0, 1, 2], [2, 1, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    original = PBRMaterial(name="wall", baseColorFactor=[255, 255, 255, 255])
    mesh.visual = TextureVisuals(uv=np.zeros((len(vertices), 2)), material=original)
    changed_material = PBRMaterial(name="wall_source_detail", baseColorFactor=[255, 255, 255, 255])
    parts = dict(_split_mesh(mesh, np.asarray([True, False]),
                             np.ones((len(vertices), 2)) * .5, changed_material, "source-detail"))
    changed, unchanged = parts["source-detail"], parts["unchanged"]
    assert changed.visual.material.name == "wall_source_detail"
    assert unchanged.visual.material is original
    assert np.array_equal(np.sort(changed.vertices, axis=0), np.sort(vertices[faces[0]], axis=0))
    assert np.array_equal(np.sort(unchanged.vertices, axis=0), np.sort(vertices[faces[1]], axis=0))


def test_normal_and_roughness_channels_can_share_reprojected_source_uvs():
    y, x = np.mgrid[:16, :16]
    normal = Image.fromarray(np.stack([x * 16, y * 16, np.full_like(x, 128)], axis=2).astype(np.uint8), "RGB")
    rough = Image.fromarray(np.stack([np.zeros_like(x), x * 16, y * 16], axis=2).astype(np.uint8), "RGB")
    h, w = 12, 20
    yy, xx = np.mgrid[:h, :w]
    uv = np.column_stack([(xx.ravel() + .5) / w, (yy.ravel() + .5) / h])
    n_atlas = np.asarray(resample_map_to_source_frame(normal, uv, (h, w)))
    r_atlas = np.asarray(resample_map_to_source_frame(rough, uv, (h, w)))
    assert n_atlas.shape == r_atlas.shape == (h, w, 3)
    # Both maps sample the same physical source-frame pixels; their channel payloads stay separate.
    assert np.array_equal(n_atlas[..., 0], r_atlas[..., 1])
    assert np.array_equal(n_atlas[..., 1], r_atlas[..., 2])


def test_reprojected_material_uses_pixel_footprint_filter_before_downsampling():
    yy, xx = np.mgrid[:32, :32]
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    texture = Image.fromarray(np.repeat(checker[..., None], 3, axis=2), "RGB")
    # Four texture texels pass beneath each atlas pixel, so the checker is below
    # the atlas Nyquist limit and should integrate toward its neutral mean.
    filtered = np.asarray(_prefilter_for_uv_footprint(texture, np.array([[4 / 32, 0], [0, 4 / 32]])))
    assert np.abs(filtered.astype(float).mean() - 127.5) < 2
    assert filtered.std() < 2

