import numpy as np
import trimesh
from shapely.geometry import Polygon

from general_world.surface_orientation import audit_downward_caps


FP = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])


def cap(z=10):
    v = [[0, 0, z], [4, 0, z], [4, 4, z], [0, 4, z]]
    return trimesh.Trimesh(v, [[0, 2, 1], [0, 3, 2]], process=False)


def lidar(z=10):
    return np.asarray([[x + .2, y + .2, z] for x, y in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
                                                            (0, 2), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]])


def test_open_downward_cap_with_supported_lidar_is_repairable():
    result = audit_downward_caps(cap(), [], FP, lidar(), 5)
    assert result["repairable"]
    assert result["reverse_face_indices"] == [0, 1]


def test_closed_solid_soffit_is_never_reversed():
    solid = trimesh.creation.box([4, 4, .2]); solid.apply_translation([2, 2, 10])
    result = audit_downward_caps(solid, [], FP, lidar(9.9), 5)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "closed-solid"


def test_upward_roof_above_candidate_skips_repair():
    upward = trimesh.creation.box([4, 4, .1]); upward.apply_translation([2, 2, 10.5])
    result = audit_downward_caps(cap(), [upward], FP, lidar(), 5)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "overlying-upward-surface"


def test_missing_or_unsupported_lidar_skips_repair():
    result = audit_downward_caps(cap(), [], FP, np.empty((0, 3)), 5)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "insufficient-distributed-lidar"


def test_connected_wall_extending_above_candidate_skips_repair():
    v = [[0, 0, 10], [4, 0, 10], [4, 0, 11], [0, 0, 11]]
    wall = trimesh.Trimesh(v, [[0, 1, 2], [0, 2, 3]], process=False)
    shell=trimesh.util.concatenate([cap(),wall])
    result = audit_downward_caps(shell, [], FP, lidar(), 5)
    assert not result["repairable"]
    assert result["reports"][0]["reason"] == "adjoining-wall-above-plane"

def test_neighboring_higher_wall_is_not_a_connected_ceiling_shell():
    wall=trimesh.Trimesh([[0,0,9],[4,0,9],[4,0,11],[0,0,11]],[[0,1,2],[0,2,3]],process=False)
    assert audit_downward_caps(cap(),[wall],FP,lidar(),5)['repairable']

def test_two_overlying_roof_halves_are_combined_before_soffit_decision():
    halves=[]
    for x in (1.,3.):
        m=trimesh.creation.box([2,4,.1]);m.apply_translation([x,2,10.5]);halves.append(m)
    assert not audit_downward_caps(cap(),halves,FP,lidar(),5)['repairable']
