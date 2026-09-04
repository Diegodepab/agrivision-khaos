from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from agrivision_khaos.dry_run import audit_raw_datasets
from agrivision_khaos.synthetic import generate_mock_dataset


class DryRunTests(unittest.TestCase):
    def test_clean_mock_dataset_validates_all_three_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            generate_mock_dataset(raw_dir, samples_per_format=4, seed=11)

            report = audit_raw_datasets(raw_dir)

            self.assertTrue(report["valid"])
            self.assertEqual(report["counts"]["sources"], 3)
            self.assertEqual(report["counts"]["images"], 12)
            self.assertEqual(report["counts"]["annotations"], 12)
            self.assertEqual(report["counts"]["duplicate_groups"], 4)
            self.assertEqual(report["counts"]["conflicting_duplicate_groups"], 0)
            self.assertEqual(
                {source["format"] for source in report["sources"]},
                {"coco", "yolo", "voc"},
            )

    def test_edge_fixture_detects_corruption_bounds_and_hash_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            generate_mock_dataset(
                raw_dir,
                samples_per_format=4,
                seed=12,
                include_edge_cases=True,
            )

            report = audit_raw_datasets(raw_dir)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertFalse(report["valid"])
            self.assertIn("corrupt_image", codes)
            self.assertIn("bbox_out_of_bounds", codes)
            self.assertIn("conflicting_duplicate_labels", codes)
            self.assertEqual(report["counts"]["conflicting_duplicate_groups"], 1)

    def test_mock_generation_is_reproducible_for_equal_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            generate_mock_dataset(first, samples_per_format=2, seed=99)
            generate_mock_dataset(second, samples_per_format=2, seed=99)

            first_hash = hashlib.sha256(
                (first / "mock_coco" / "train" / "coco-00.jpg").read_bytes()
            ).hexdigest()
            second_hash = hashlib.sha256(
                (second / "mock_coco" / "train" / "coco-00.jpg").read_bytes()
            ).hexdigest()

            self.assertEqual(first_hash, second_hash)

    def test_generator_refuses_to_overwrite_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "raw"
            output.mkdir()
            (output / "owned-by-user.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                generate_mock_dataset(output)

            self.assertEqual(
                (output / "owned-by-user.txt").read_text(encoding="utf-8"),
                "keep",
            )


if __name__ == "__main__":
    unittest.main()
