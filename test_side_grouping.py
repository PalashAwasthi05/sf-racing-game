import itertools
import math
import unittest

from regen.side_grouping import sides_of


def wall(name, start, end, offset=0., angle=0., height=5.):
    theta = math.radians(angle)
    return {'id': name, 'normal': [math.sin(theta), -math.cos(theta)],
            'midpoint': [(start + end) / 2, offset],
            'lengthM': end - start, 'heightM': height}


class SideGroupingTests(unittest.TestCase):
    def test_offset_parallel_facades_remain_separate(self):
        self.assertEqual(len(sides_of([wall('a', 0, 10), wall('b', 10, 20, offset=.3)])), 2)

    def test_angled_facade_remains_separate(self):
        self.assertEqual(len(sides_of([wall('a', 0, 10), wall('b', 10, 20, angle=5)])), 2)

    def test_both_endpoints_must_fit_common_plane(self):
        # Midpoint is on the anchor plane and angle < 1 degree, but ends are not.
        self.assertEqual(len(sides_of([wall('a', 0, 100), wall('b', 100, 150, angle=.8)])), 2)

    def test_disjoint_coplanar_segments_remain_separate(self):
        self.assertEqual(len(sides_of([wall('a', 0, 10), wall('b', 10.26, 20)])), 2)

    def test_small_gap_is_allowed(self):
        self.assertEqual(len(sides_of([wall('a', 0, 10), wall('b', 10.25, 20)])), 1)

    def test_contiguous_bridge_is_revisited_regardless_of_input_order(self):
        walls = [wall('anchor', 0, 10), wall('far', 12, 20, height=9), wall('bridge', 10, 12)]
        for order in itertools.permutations(walls):
            with self.subTest(order=[w['id'] for w in order]):
                groups = sides_of(order)
                self.assertEqual(len(groups), 1)
                self.assertEqual((groups[0]['u0'], groups[0]['u1'], groups[0]['h']), (0, 20, 9))
                self.assertEqual({w['id'] for w in groups[0]['walls']}, {'anchor', 'far', 'bridge'})
                self.assertTrue(all(any(w is original for original in walls) for w in groups[0]['walls']))

    def test_offsets_cannot_drift_from_anchor_plane(self):
        self.assertEqual(len(sides_of([wall('a', 0, 10), wall('b', 10, 19, offset=.1),
                                       wall('c', 19, 27, offset=.2)])), 2)


if __name__ == '__main__':
    unittest.main()
