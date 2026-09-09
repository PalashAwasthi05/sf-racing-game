import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box

from general_world.member_support import clip_long_members


def _strip(bounds):
    lo, hi = bounds
    return trimesh.creation.box([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
                                transform=trimesh.transformations.translation_matrix((np.array(lo) + hi) / 2))


def test_unsupported_strip_crossing_concave_footprint_is_clipped():
    footprint = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)])
    strip = _strip(((-10, 3, 10), (25, 4, 10.2)))
    replacements, report = clip_long_members(strip, footprint, missingSupportEvidence=True)
    assert report["offending"] == 1 and report["changed"]
    clipped = replacements[0]["mesh"]
    assert clipped.is_volume and clipped.volume < strip.volume
    assert all(footprint.buffer(2.0).covers(Point(float(x), float(y))) for x, y in clipped.vertices[:, :2])


def test_supported_strip_is_retained_when_overreach_is_small():
    footprint = box(0, 0, 20, 10)
    strip = _strip(((-1, 4, 10), (21, 5, 10.2)))
    replacements, report = clip_long_members(strip, footprint, missingSupportEvidence=True)
    assert not report["changed"] and report["offending"] == 0
    assert replacements == []


def test_measured_cantilever_is_exempt():
    footprint = box(0, 0, 20, 10)
    strip = _strip(((-10, 4, 10), (30, 5, 10.2)))
    replacements, report = clip_long_members(strip, footprint, missingSupportEvidence=True, measuredSupport=True)
    assert replacements == []
    assert report == {"changed": False, "reason": "measured-support", "components": 0, "offending": 0}


def test_missing_support_evidence_is_required():
    footprint = box(0, 0, 20, 10)
    strip = _strip(((-10, 4, 10), (30, 5, 10.2)))
    replacements, report = clip_long_members(strip, footprint)
    assert replacements == []
    assert report["reason"] == "missing-support-evidence-required"


def test_rotated_strip_and_unrelated_component_do_not_change_detection():
    from shapely.affinity import rotate
    footprint=rotate(box(0,0,20,10),45,origin=(0,0))
    strip=_strip(((-10,4,10),(30,5,10.2)));strip.apply_transform(trimesh.transformations.rotation_matrix(np.pi/4,[0,0,1]))
    unrelated=_strip(((100,100,0),(101,101,1)))
    _,r=clip_long_members(trimesh.util.concatenate([strip,unrelated]),footprint,missingSupportEvidence=True)
    assert r['offending']==1 and r['changed']

def test_disconnected_support_polygons_are_unioned():
    from shapely.ops import unary_union
    footprint=unary_union([box(0,0,5,10),box(15,0,20,10)])
    strip=_strip(((-10,4,10),(30,5,10.2)))
    parts,r=clip_long_members(strip,footprint,margin_m=0,missingSupportEvidence=True)
    assert r['changed'] and r['failedClips']==0
    assert parts[0]['mesh'].volume==__import__('pytest').approx(2,abs=.001)
