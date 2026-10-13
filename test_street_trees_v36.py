"""Curb-side street trees move a wrongly modelled curb, and repeated inventory records are one tree
(pipeline/street_space.curb_from_trees, tree_candidates). Synthetic geometry only."""
import json, sys, pathlib
from shapely.geometry import box, Point
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import street_space as S


def ctx_with(trees, tmp):
    (tmp / 'trees.json').write_text(json.dumps(trees))
    return {'DATA': tmp, 'json': json, 'local': lambda lon, lat: (lon, lat)}


def test_a_row_of_curb_side_trees_inside_the_road_moves_the_curb(tmp_path):
    road = box(0, 0, 100, 10); sidewalk = box(0, 10, 100, 14); crossing = box(90, 0, 94, 14)
    trees = [{'longitude': x, 'latitude': 8.5, 'qsiteinfo': 'Sidewalk: Curb side : Cutout', 'qspecies': 'Washingtonia robusta'} for x in (20, 28, 36)]
    r2, s2, ev = S.curb_from_trees(ctx_with(trees, tmp_path), road, sidewalk, crossing)
    assert len(ev) == 3 and all(s2.covers(Point(x, 8.5)) for x in (20, 28, 36))
    assert not r2.covers(Point(28, 8.2)) and r2.covers(Point(28, 7.)) and r2.covers(Point(60, 9.))   # only along the run


def test_a_lone_tree_or_one_far_inside_the_road_does_not(tmp_path):
    road = box(0, 0, 100, 10); sidewalk = box(0, 10, 100, 14); crossing = box(90, 0, 94, 14)
    trees = [{'longitude': 20, 'latitude': 8.5, 'qsiteinfo': 'Sidewalk: Curb side : Cutout', 'qspecies': 'x'},
             {'longitude': 70, 'latitude': 3., 'qsiteinfo': 'Sidewalk: Curb side : Cutout', 'qspecies': 'x'},
             {'longitude': 74, 'latitude': 3., 'qsiteinfo': 'Median : Yard', 'qspecies': 'x'}]
    r2, s2, ev = S.curb_from_trees(ctx_with(trees, tmp_path), road, sidewalk, crossing)
    assert ev == [] and r2.equals(road)

