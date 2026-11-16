import unittest

from regen.specs import to_profile


def piece():
    return {
        "id": "source-a",
        "building": "osm_123_p0",
        "lengthM": 10.,
        "heightM": 8.,
        "normal": [1., 0.],
        "frameU": [100., 110.],
        "planeV": 20.,
        "views": [{"image": "view-a"}],
    }


def reading(ornaments):
    return {
        "confidence": "high",
        "materials": [{"family": "stone", "hex": "#999999", "x0": 0., "x1": 10., "z0": 0., "z1": 8.}],
        "ornaments": ornaments,
    }


class OrnamentShapeSpecTests(unittest.TestCase):
    def test_explicit_shape_and_typed_repeat_fields_reach_relief(self):
        profile = to_profile(reading([{
            "x0": 2., "x1": 4., "z0": 3., "z1": 4., "shape": "roundel",
            "repeatCount": 5, "ringCount": 2, "curveDepth": .42,
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertEqual(relief["shape"], "roundel")
        self.assertEqual(relief["repeatCount"], 5)
        self.assertEqual(relief["ringCount"], 2)
        self.assertAlmostEqual(relief["curveDepth"], .42)
        self.assertEqual(relief["rect"], [.2, .375, .4, .5])

    def test_other_supported_shapes_are_preserved(self):
        ornaments = [
            {"x0": 0., "x1": 1., "z0": 1., "z1": 2., "shape": "cartouche"},
            {"x0": 2., "x1": 3., "z0": 1., "z1": 2., "shape": "garland"},
            {"x0": 4., "x1": 5., "z0": 1., "z1": 2., "shape": "panel"},
        ]
        profile = to_profile(reading(ornaments), piece())
        self.assertEqual([r["shape"] for r in profile["reliefs"]], ["cartouche", "garland", "panel"])

    def test_unknown_shape_defaults_to_panel_without_kind_guessing(self):
        profile = to_profile(reading([{
            "x0": 1., "x1": 2., "z0": 1., "z1": 2., "shape": "medallion", "kind": "roundel",
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertEqual(relief["shape"], "panel")
        self.assertEqual(relief["repeatCount"], 1)
        self.assertNotIn("kind", relief)

    def test_optional_counts_and_curve_depth_are_bounded(self):
        profile = to_profile(reading([{
            "x0": 1., "x1": 2., "z0": 1., "z1": 2., "shape": "cartouche",
            "repeatCount": 99, "ringCount": 0, "curveDepth": 1.7,
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertEqual(relief["repeatCount"], 20)
        self.assertEqual(relief["ringCount"], 1)
        self.assertEqual(relief["curveDepth"], 1.)

    def test_invalid_optional_fields_are_omitted(self):
        profile = to_profile(reading([{
            "x0": 1., "x1": 2., "z0": 1., "z1": 2., "shape": "roundel",
            "repeatCount": "many", "ringCount": "three", "curveDepth": "deep",
            "outerOutline": "diamond", "motif": "scroll", "cardinalFlourishes": "yes",
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertEqual(relief["repeatCount"], 1)
        self.assertNotIn("ringCount", relief)
        self.assertNotIn("curveDepth", relief)
        self.assertNotIn("outerOutline", relief)
        self.assertNotIn("motif", relief)
        self.assertNotIn("cardinalFlourishes", relief)

    def test_relief_outline_motif_and_cardinal_flourishes_are_preserved_independently(self):
        profile = to_profile(reading([{
            "x0": 2., "x1": 4., "z0": 3., "z1": 4., "shape": "cartouche",
            "outerOutline": "circle", "motif": "radial", "cardinalFlourishes": True,
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertEqual(relief["shape"], "cartouche")  # legacy field remains compatible
        self.assertEqual(relief["outerOutline"], "circle")
        self.assertEqual(relief["motif"], "radial")
        self.assertIs(relief["cardinalFlourishes"], True)

    def test_missing_relief_semantics_remain_absent(self):
        profile = to_profile(reading([{
            "x0": 2., "x1": 4., "z0": 3., "z1": 4., "shape": "roundel",
        }]), piece())
        relief = profile["reliefs"][0]
        self.assertFalse({'outerOutline', 'motif', 'cardinalFlourishes'} & relief.keys())


if __name__ == "__main__":
    unittest.main()
