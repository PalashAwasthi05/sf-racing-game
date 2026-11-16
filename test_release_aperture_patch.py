import numpy as np
import trimesh

from regen.glb_stream import GlbAppendWriter, GltfStream
from regen.release_aperture_patch import _postwrite_invariants


def test_serialized_append_preserves_base_binary_materials_and_node_poses(tmp_path):
    base = tmp_path / 'base.glb'
    candidate = tmp_path / 'candidate.glb'
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name='buildings__fixture___shell',
                       geom_name='fixture-shell')
    scene.export(base)

    with GltfStream(base) as stream:
        writer = GlbAppendWriter(stream)
        writer.add_node('facades__fixture__aperture_receiver_0_glazing',
                        trimesh.creation.box(extents=[.2, .2, .02]), None,
                        np.eye(4), {'featureId': 'fixture', 'evidence': 'source-measured-aperture-repair'})
        writer.write(candidate)
        writer.close()

    report = {'outcomes': [{'featureId': 'fixture', 'status': 'staged'}],
              'auditConflicts': []}
    checks = _postwrite_invariants(base, candidate, report)
    assert checks['allInvariantsPassed']
    assert checks['originalBinaryPrefixSha256']
    assert checks['originalMaterialsUnchanged']
    assert checks['imagesAndTexturesUnchanged']
    assert checks['originalCamerasUnchanged']
    assert checks['originalNodePoseAndPropertiesPreserved']
    assert checks['replacedNodeMaterialAssignmentsPreserved']
    assert checks['originalSceneRootsPreserved']
