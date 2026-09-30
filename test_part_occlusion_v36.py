"""Parts of one split outline occlude each other (frontage_parts_v36) and a band read over a building's own lower part can
speak for the part's material (facades.band_agreed). Synthetic geometry only."""
import sys, pathlib
import numpy as np
from shapely.geometry import box
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontage_parts_v36 as FP
from pipeline import facades


class _Occ:
    def __init__(self, g): self.g = g


def _own_footprints(wall, occ):
    return occ.g


def test_own_mass_leaves_out_the_other_part_but_keeps_outline_slack():
    outline = box(0, 0, 30, 22)                      # the source footprint runs 2 m in front of both parts
    low, tall = box(0, 2, 12, 12), box(0, 12, 30, 22).union(box(12, 2, 30, 12))
    parts = {'b': tall, 'b_p1': low}
    wall_tall = {'buildingId': 'b', 'midpoint': [6., 12.], 'normal': [0., -1.]}
    own = FP.own_mass(wall_tall, _Occ(outline), parts, _own_footprints)
    assert not own.contains(box(2, 4, 10, 10).centroid)          # the low building is not the tall block's own mass
    assert own.contains(box(0, 0, 30, 1.5).centroid)             # the slack strip in front stays own
    wall_low = {'buildingId': 'b_p1', 'midpoint': [6., 2.], 'normal': [0., -1.]}
    own_low = FP.own_mass(wall_low, _Occ(outline), parts, _own_footprints)
    assert own_low.contains(box(2, 4, 10, 10).centroid) and not own_low.contains(box(14, 14, 28, 20).centroid)
    solo = {'buildingId': 'c', 'midpoint': [0., 0.], 'normal': [0., -1.]}
    assert FP.own_mass(solo, _Occ(outline), parts, _own_footprints) is outline   # an unsplit building keeps its footprints


def test_raised_surface_stands_the_other_part_to_its_top():
    st = 2.; top = np.full((11, 15), np.nan, np.float32)            # raster over x 0..30, y 22..0 (row 0 at y 22)
    top[5:10, 0:6] = np.linspace(5., 7., 5)[:, None]              # the low part, a sloped roof read 5-7 m
    surf = (top, (0., 22., st))
    parts = {'b': box(0, 12, 30, 22), 'b_p1': box(0, 2, 12, 12)}
    raised = FP.raised_surface(surf, {'buildingId': 'b'}, parts)[0]
    assert np.nanmin(raised[5:10, 0:6]) >= 7. + .9                # every cell at the part's top plus the margin
    same = FP.raised_surface(surf, {'buildingId': 'b_p1'}, {'b_p1': parts['b_p1']})
    assert same is surf                                            # no other part: the raster as measured


def test_band_agreed_needs_the_band_mark_and_two_photographs():
    band = {'reading': {'aboveOwnPart': .12, 'visibleShare': .12, 'consensus': {'agreeing': 2}}}
    assert facades.band_agreed(band)
    assert not facades.band_agreed({'reading': {'aboveOwnPart': .12, 'consensus': {'agreeing': 1}}})
    assert not facades.band_agreed({'reading': {'visibleShare': .12, 'consensus': {'agreeing': 5}}})   # trees or cars, not a lower part


def test_parapet_solid_cuts_a_wall_to_its_measured_steps():
    import manifold3d
    r = {'a': [0., 0.], 'b': [10., 0.], 'normal': [0., -1.], 'z0': 2., 'top': [[0., 6.], [3., 6.], [3., 7.], [7., 7.], [7., 6.], [10., 6.]]}
    wall = manifold3d.Manifold.cube([10., .4, 9.]).translate([0., -.2, 2.])      # a 9 m wall on the line a-b
    cut = wall ^ facades.parapet_solid(r, margin_m=0.)
    # 6 m of wall over its 10 m length plus the 1 m step over its middle 4 m, 0.4 m thick; nothing above the step
    assert abs(cut.volume() - .4 * (6. * 10. + 1. * 4.)) < 1e-6
    assert abs(cut.bounding_box()[5] - 9.) < 1e-6


def test_measured_parapet_matches_the_wall_along_it(monkeypatch):
    rec = {'buildingId': 'b', 'a': [0., 0.], 'b': [10., 0.], 'normal': [0., -1.], 'z0': 0., 'top': [[0., 5.], [10., 5.]]}
    monkeypatch.setattr(facades, '_PARAPETS', [rec])
    assert facades.measured_parapet('b', [.2, .3], [9.8, .3]) is rec
    assert facades.measured_parapet('b', [0., 5.], [10., 5.]) is None      # a parallel wall 5 m away
    assert facades.measured_parapet('c', [.2, .3], [9.8, .3]) is None


def test_a_pier_read_near_black_in_the_walls_hue_is_the_wall_in_shade():
    from pipeline.colour_agreement import shaded_wall
    assert shaded_wall([70, 33, 31], [178, 84, 60])            # a brick pilaster's shaded side
    assert not shaded_wall([20, 20, 20], [112, 124, 138])      # a black pier on a grey wall
    assert not shaded_wall([63, 41, 36], [124, 134, 142])      # dark brown on a grey wall: another material
    assert not shaded_wall([150, 70, 50], [178, 84, 60])       # a lit brick pier is not shade
