"""v33: a facing's reading is the one its photographs agree on (auto_read_frontages_v33.consensus / pick_photos)."""
import numpy as np
import auto_read_frontages_v33 as C


def reading(rgb, windows=0, visible=.9, material_only=False):
    p = {'materialEvidence': {'colorRgb': rgb}, 'reading': {'visibleShare': visible}}
    if material_only: p['materialOnly'] = True
    else: p['reading'].update(storeys=max(1, windows // 4), windows=windows)
    return ({'image': str(len(rgb) + windows)}, p)


def test_an_outlier_photograph_is_outvoted_and_the_agreeing_group_keeps_its_best_viewed_reading():
    # best-scored photograph reads grey (a neighbour through a pose error); two others agree on the pink brick
    rs = [reading([119, 115, 109], 6), reading([219, 151, 130], 12), reading([214, 146, 124], 15)]
    k, agree = C.consensus(rs)
    assert agree == 2 and k == 1                         # the pink group; within it the best-viewed photograph


def test_a_near_black_reading_is_set_aside_while_another_exists():
    rs = [reading([10, 8, 4], 20), reading([150, 90, 70], 4)]
    assert C.consensus(rs)[0] == 1
    assert C.consensus([reading([10, 8, 4], 20)])[0] == 0           # alone, it is all there is


def test_without_agreement_the_best_scored_reading_is_kept_and_flagged():
    rs = [reading([119, 115, 109], 6), reading([219, 151, 130], 12), reading([60, 90, 140], 15)]
    k, agree = C.consensus(rs)
    assert k == 0 and agree == 1


def test_photographs_of_one_drive_count_once_unless_far_apart():
    cams = {'a': {'captureDate': '2024-05-01', 'xy': np.array([0., 0.])}, 'b': {'captureDate': '2024-05-01', 'xy': np.array([4., 0.])},
            'c': {'captureDate': '2024-05-01', 'xy': np.array([14., 0.])}, 'd': {'captureDate': '2023-02-11', 'xy': np.array([1., 0.])},
            'e': {'captureDate': '2025-01-09', 'xy': np.array([0., 1.])}}
    cands = [{'image': i, 'score': s} for i, s in (('a', 1.), ('b', .95), ('c', .9), ('d', .5), ('e', .3))]
    picked = [c['image'] for c in C.pick_photos(cands, cams, 5)]
    assert picked == ['a', 'c', 'd']                    # b: same drive 4 m away; e: under 40 % of the best score


def test_agreement_does_not_chain_through_an_intermediate_colour():
    # pale pink ~ rose ~ deep rose step by step (dE 8.7, 12.7) but pale pink and deep rose disagree (dE 21): only pairs agree
    rs = [reading([226, 176, 170], 22), reading([205, 145, 150], 0), reading([180, 110, 135], 12), reading([106, 65, 60], 7)]
    k, agree = C.consensus(rs)
    assert agree == 2 and k == 0                         # the pair holding the best-scored photograph
    assert C.agreeing([0, 1, 2], lambda i, j: abs(i - j) <= 1) == [0, 1]


def test_an_established_reading_stands_unless_a_larger_set_agrees_against_it():
    inc = reading([217, 145, 123], 15)                  # the published reading, read first
    # one other photograph agrees with it, two agree with each other on something else: a tie keeps the incumbent
    rs = [inc, reading([119, 115, 109], 16), reading([219, 151, 130], 13), reading([124, 118, 110], 30)]
    assert C.consensus(rs, keep_first=True)[0] == 0
    # three others agree against it and none with it: outvoted
    rs = [inc, reading([119, 115, 109], 16), reading([122, 117, 111], 13), reading([124, 118, 110], 30)]
    assert C.consensus(rs, keep_first=True)[0] == 1
    # a pair agreeing against a lone incumbent does not overturn it
    rs = [inc, reading([119, 115, 109], 16), reading([122, 117, 111], 13), reading([60, 90, 140], 30)]
    assert C.consensus(rs, keep_first=True)[0] == 0
    # a near-black incumbent yields to two lit photographs that agree, not to one
    rs = [reading([10, 8, 4], 20), reading([150, 90, 70], 4), reading([146, 92, 72], 6)]
    assert C.consensus(rs, keep_first=True)[0] == 1
    rs = [reading([10, 8, 4], 20), reading([150, 90, 70], 4), reading([60, 90, 140], 6)]
    assert C.consensus(rs, keep_first=True)[0] == 0


def test_the_incumbent_photograph_is_read_first_even_when_others_score_better():
    cams = {'a': {'captureDate': '2024-05-01', 'xy': np.array([0., 0.])}, 'z': {'captureDate': '2021-03-02', 'xy': np.array([30., 0.])}}
    cands = [{'image': 'a', 'score': 1.}, {'image': 'z', 'score': .2}]
    assert [c['image'] for c in C.pick_photos(cands, cams, 3, first='z')] == ['z', 'a']
