import numpy as np
import trimesh
from pipeline.mesh_attributes import transfer_metric_uv


def test_oblique_clip_preserves_metric_chart_at_new_and_surviving_vertices():
    mesh = trimesh.creation.box(extents=[3., .12, 4.])
    # A deliberately non-world-unit chart: replacing it with automatic projection
    # would change both phase and metric repeat size after clipping.
    mesh.metadata['uvMeters'] = np.c_[mesh.vertices[:,0]*1.7+11.3, mesh.vertices[:,2]+7.2]
    clipped = trimesh.intersections.slice_mesh_plane(mesh, [-1., 0., -1.], [.4, 0., .4], cap=False)
    clipped.metadata.clear()  # Boolean/slice APIs may retain stale original metadata.
    result = transfer_metric_uv(mesh, clipped)
    expected = np.c_[result.vertices[:,0]*1.7+11.3, result.vertices[:,2]+7.2]
    np.testing.assert_allclose(result.metadata['uvMeters'], expected, atol=1e-8)


def test_surface_without_authored_uv_remains_unchanged_and_empty_clip_is_safe():
    mesh = trimesh.creation.box()
    target = mesh.copy()
    assert transfer_metric_uv(mesh, target) is target
    assert 'uvMeters' not in target.metadata
    mesh.metadata['uvMeters'] = np.c_[mesh.vertices[:,0], mesh.vertices[:,2]]
    assert transfer_metric_uv(mesh, None) is None
    assert transfer_metric_uv(mesh, mesh) is mesh
