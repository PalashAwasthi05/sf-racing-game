import unittest
import numpy as np

from general_world.material_periodicity import infer_joint_periods


class MaterialPeriodicityTests(unittest.TestCase):
    def test_repeated_mortar_rows_and_columns(self):
        rng = np.random.default_rng(7)
        image = np.full((192, 192), 0.72, dtype=float)
        image[::24, :] = 0.12
        image[:, ::32] = 0.12
        image += rng.normal(0, 0.025, image.shape)
        result = infer_joint_periods(image, min_period_px=6)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["horizontal"]["period_px"], 24)
        self.assertEqual(result["vertical"]["period_px"], 32)
        self.assertGreater(result["horizontal"]["confidence"], 0.4)

    def test_smooth_grain_is_rejected(self):
        y, x = np.mgrid[:192, :192]
        image = 0.5 + 0.08 * np.sin(x / 17.0) + 0.06 * np.sin(y / 21.0)
        result = infer_joint_periods(image, min_period_px=6)
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["horizontal"]["period_px"])
        self.assertIsNone(result["vertical"]["period_px"])

    def test_running_bond_reports_full_module_and_stagger(self):
        rng = np.random.default_rng(8)
        image = np.full((192, 256), 0.78, dtype=float)
        row_pitch, full = 24, 64
        for row in range(0, image.shape[0], row_pitch):
            image[row:row + row_pitch, :] = 0.72
            offset = 0 if (row // row_pitch) % 2 == 0 else full // 2
            image[row:row + row_pitch, offset::full] = 0.10
            image[row:row + 2, :] = 0.10
        image += rng.normal(0, 0.02, image.shape)
        result = infer_joint_periods(image, min_period_px=6)
        bond = result["banded_running_bond"]
        self.assertTrue(bond["accepted"])
        self.assertEqual(bond["row_pitch_px"], row_pitch)
        self.assertEqual(bond["full_module_period_px"], full)
        self.assertTrue(bond["stagger_consistent"])


if __name__ == "__main__":
    unittest.main()
