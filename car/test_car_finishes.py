"""The generator and the viewer agree on finish names (an unknown finish would silently fall back to black trim)."""
import os
import re
import build_roadster as B

CAR_TS = os.path.join(os.path.dirname(__file__), '..', 'king-street', 'app', 'car.ts')


def viewer_finishes():
    src = open(CAR_TS).read()
    block = src[src.index('function finishes()'):src.index('type Finish=')]
    return set(re.findall(r'^\s{2}(\w+):(?:phys|std)\(', block, re.M))


def test_every_generated_finish_has_a_viewer_material():
    assert set(B.FINISHES) <= viewer_finishes()


def test_viewer_declares_no_unused_finishes():
    assert viewer_finishes() <= set(B.FINISHES)
