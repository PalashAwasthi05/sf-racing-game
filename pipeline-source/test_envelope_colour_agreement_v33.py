"""Envelope parts take a photographed colour only when two or more photographs agree on it (pipeline/envelopes.py)."""
from shapely.geometry import box
from pipeline import envelopes as E


def cells(rgb, w=10., h=10.):
    return [[x, z, x + 2., z + 2., rgb, 1.] for x in range(0, int(w), 2) for z in range(0, int(h), 2)]


def test_two_agreeing_photographs_outvote_a_third():
    face = box(0, 0, 10, 10)
    obs = {'photos': [{'camera': 'a', 'cells': cells([112, 60, 50])}, {'camera': 'b', 'cells': cells([96, 104, 88])}, {'camera': 'c', 'cells': cells([118, 64, 52])}]}
    assert E.observed_colour('brick_modular', obs, face) == 'brick_modular#733e33'     # median of the two brick readings


def test_one_photograph_or_disagreeing_photographs_leave_the_designed_colour():
    face = box(0, 0, 10, 10)
    assert E.observed_colour('brick_modular', {'cells': cells([96, 104, 88])}, face) == 'brick_modular'
    obs = {'photos': [{'camera': 'a', 'cells': cells([112, 60, 50])}, {'camera': 'b', 'cells': cells([96, 104, 88])}]}
    assert E.observed_colour('brick_modular', obs, face) == 'brick_modular'


def test_a_photograph_that_sees_too_little_of_the_part_does_not_vote():
    face = box(0, 0, 10, 10)
    obs = {'photos': [{'camera': 'a', 'cells': cells([112, 60, 50])}, {'camera': 'b', 'cells': cells([110, 62, 50], w=2., h=2.)}]}
    assert E.observed_colour('brick_modular', obs, face) == 'brick_modular'           # b covers 4 % of the face
