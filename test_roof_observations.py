import unittest

from pipeline.roof_observations import proposal


def profile(part="p1", feature="f1", roofs=(), confidence="high", **extra):
    return {"match": {"partId": part, "featureId": feature},
            "reading": {"confidence": confidence}, "roofs": list(roofs), **extra}


class RoofObservationTests(unittest.TestCase):
    def test_preserves_observed_dimensions(self):
        p = profile(roofs=[{"kind": "mansard", "fullWidth": True, "pitchDeg": 67,
                            "riseM": 2.3, "overhangM": .45}])
        spec, audit = proposal([p], "p1", "f1")
        self.assertEqual(spec, {"kind": "mansard", "pitchDeg": 67, "riseM": 2.3, "overhangM": .45})
        self.assertEqual(audit["status"], "proposed")
        self.assertEqual(audit["provenance"][0]["partId"], "p1")

    def test_partial_facade_roof_does_not_become_building_roof(self):
        p = profile(roofs=[{"kind": "hip", "widthShare": .35, "riseM": 1.4}])
        spec, audit = proposal([p], "p1", "f1")
        self.assertIsNone(spec)
        self.assertEqual(audit["status"], "no-global-roof")
        self.assertEqual(audit["excluded"][0]["reason"], "facade-local-or-partial")

    def test_conflicting_global_kinds_are_unresolved(self):
        ps = [profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True, "riseM": 2}]),
              profile(part="p1", feature="f1", roofs=[{"kind": "hip", "fullWidth": True, "riseM": 2}])]
        spec, audit = proposal(ps, "p1", "f1")
        self.assertIsNone(spec)
        self.assertEqual(audit["conflicts"], ["conflicting-kinds"])

    def test_gable_outline_supports_explicit_gable(self):
        p = profile(roofs=[{"kind": "gable", "riseM": 2.1}], gableOutline=True)
        spec, _ = proposal([p], "p1", "f1")
        self.assertEqual(spec, {"kind": "gable", "riseM": 2.1})

    def test_joins_by_ids_and_ignores_nearby_profiles(self):
        exact = profile(part="part-1", feature="source-1", roofs=[{"kind": "shed", "fullWidth": True}])
        unrelated = profile(part="part-10", feature="source-10", roofs=[{"kind": "hip", "fullWidth": True}])
        spec, audit = proposal([unrelated, exact], "part-1", "source-1")
        self.assertEqual(spec, {"kind": "shed"})
        self.assertEqual(len(audit["provenance"]), 1)

    def test_normalized_x_span_is_full_width_but_local_scope_overrides(self):
        p = profile(roofs=[{"kind": "hip", "x": [0.05, .96], "riseM": 1.2}])
        spec, _ = proposal([p], "p1", "f1")
        self.assertEqual(spec, {"kind": "hip", "riseM": 1.2})
        p["roofs"][0]["scope"] = "dormer"
        spec, audit = proposal([p], "p1", "f1")
        self.assertIsNone(spec)
        self.assertEqual(audit["status"], "no-global-roof")

    def test_part_id_prevents_feature_id_from_joining_sibling_part(self):
        target = profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True}])
        sibling = profile(part="p2", feature="f1", roofs=[{"kind": "hip", "fullWidth": True}])
        spec, audit = proposal([target, sibling], "p1", "f1")
        self.assertEqual(spec, {"kind": "gable"})
        self.assertEqual(len(audit["provenance"]), 1)

    def test_conflicting_dimensions_are_unresolved(self):
        ps = [profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True, "pitchDeg": 35}]),
              profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True, "pitchDeg": 42}])]
        spec, audit = proposal(ps, "p1", "f1")
        self.assertIsNone(spec)
        self.assertIn("conflicting-pitchDeg", audit["conflicts"])

    def test_small_dimension_differences_are_accepted_and_preserved(self):
        ps = [profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True, "pitchDeg": 35, "riseM": 2.0}]),
              profile(part="p1", feature="f1", roofs=[{"kind": "gable", "fullWidth": True, "pitchDeg": 36.5, "riseM": 2.15}])]
        spec, audit = proposal(ps, "p1", "f1")
        self.assertEqual(spec, {"kind": "gable", "pitchDeg": 35, "riseM": 2.0})
        self.assertEqual(audit["status"], "proposed")


if __name__ == "__main__":
    unittest.main()
def test_metric_roofline_without_pixel_provenance_is_not_a_hard_constraint():
    from pipeline.roof_observations import measured_frontage
    line, audit = measured_frontage({'pointsM': [[0, 5], [4, 8], [8, 5]], 'confidence': 'high'})
    assert line is None
    assert audit['status'] == 'unverified-source'

