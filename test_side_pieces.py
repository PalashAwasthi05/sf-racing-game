import json
import tempfile
import unittest
from pathlib import Path

from regen.side_pieces import load_side_pieces


class SidePiecesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, pieces):
        (self.root / name).write_text(json.dumps({'pieces': pieces}))

    def test_base_without_optional_tiers(self):
        base = {'id': 'base', 'frameU': [0, 10]}
        self.write('sides.json', [base])
        self.assertEqual(load_side_pieces(self.root), [base])

    def test_merge_preserves_frames_and_deduplicates_identical_pieces(self):
        base = {'id': 'base', 'frameU': [0, 10]}
        tier = {'id': 'tier', 'frameU': [2, 8], 'baseM': 12}
        self.write('sides.json', [base])
        self.write('sides-tiers.json', [base, tier])
        self.assertEqual(load_side_pieces(self.root), [base, tier])

    def test_conflicting_ids_across_files_name_both_sources(self):
        self.write('sides.json', [{'id': 'same', 'frameU': [0, 10]}])
        self.write('sides-tiers.json', [{'id': 'same', 'frameU': [1, 10]}])
        with self.assertRaisesRegex(ValueError, "Conflicting side piece ID 'same'.*sides.json.*sides-tiers.json"):
            load_side_pieces(self.root)

    def test_conflicting_ids_inside_one_file_are_rejected(self):
        self.write('sides.json', [{'id': 'same', 'baseM': 0}, {'id': 'same', 'baseM': 5}])
        with self.assertRaisesRegex(ValueError, r'pieces\[0\].*pieces\[1\]'):
            load_side_pieces(self.root)

    def test_base_file_is_required(self):
        self.write('sides-tiers.json', [{'id': 'tier'}])
        with self.assertRaises(FileNotFoundError):
            load_side_pieces(self.root)


if __name__ == '__main__':
    unittest.main()
