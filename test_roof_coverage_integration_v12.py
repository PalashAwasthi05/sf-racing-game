import numpy as np
import trimesh
from shapely.geometry import box
from general_world.roof_coverage import recover_roof_coverage

def test_existing_overhang_domain_is_never_covered_by_repair():
    mesh=trimesh.creation.extrude_polygon(box(0,0,2,4),.1,engine='earcut');mesh.apply_translation([0,0,9.9])
    points=np.array([[x,y,10.] for x in (2.2,2.7,3.2,3.7) for y in (.2,1.2,2.2,3.2)])
    made,report=recover_roof_coverage([mesh],box(0,0,4,4),points,5.,occupied_domains=[box(2,0,4,4)])
    assert not made and report['gapAreaM2']==0

def test_separate_cells_do_not_pad_evidence_for_one_near_cell():
    mesh=trimesh.creation.extrude_polygon(box(0,0,2,4),.1,engine='earcut');mesh.apply_translation([0,0,9.9])
    points=np.array([[2.2,.2,10.],[9.2,2.2,10.],[9.2,3.2,10.]])
    made,report=recover_roof_coverage([mesh],box(0,0,10,4),points,5.)
    assert not made

def test_lower_roof_does_not_mask_gap_but_higher_neighbor_does():
    mesh=trimesh.creation.extrude_polygon(box(0,0,2,4),.1,engine='earcut');mesh.apply_translation([0,0,9.9])
    points=np.array([[x,y,10.] for x in (2.2,2.7,3.2,3.7) for y in (.2,1.2,2.2,3.2)])
    made,_=recover_roof_coverage([mesh],box(0,0,4,4),points,5.,occupied_surfaces=[(box(2,0,4,4),8.)])
    assert made
    blocked,_=recover_roof_coverage([mesh],box(0,0,4,4),points,5.,occupied_surfaces=[(box(2,0,4,4),12.)])
    assert not blocked
