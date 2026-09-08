from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import cv2
import numpy as np

from agrivision_khaos.augmentation import (
    DescriptorCache,
    analyze_images,
    candidate_pairs,
    corner_padding,
)
from agrivision_khaos.augmentation_benchmark import (
    evaluate_manifest,
    generate_fixture,
    read_manifest,
)
from agrivision_khaos.families import audit_assignments, relation_groups
from agrivision_khaos.models import DeduplicationPolicy
from agrivision_khaos.split_audit import audit_visual_splits


class AugmentationTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = generate_fixture(self.root / "fixture", families=1)
        self.rows = json.loads(manifest.read_text())["samples"]
        self.paths = {row["id"]: str(manifest.parent / row["filepath"]) for row in self.rows}
        self.cache = DescriptorCache(self.root / "cache")
        self.policy = DeduplicationPolicy()

    def descriptor(self, suffix):
        return self.cache.describe(self.paths[f"leaf-0-{suffix}"])

    def test_all_quarter_turns_and_mirrors_are_full_pixel_equivalents(self):
        original = self.descriptor("r0")
        for suffix in (
            "r90",
            "r180",
            "r270",
            "mirror_r0",
            "mirror_r90",
            "mirror_r180",
            "mirror_r270",
        ):
            with self.subTest(suffix=suffix):
                evidence = self.cache.verify(original, self.descriptor(suffix), self.policy)
                self.assertEqual(evidence["level"], "exact_pixels")
                self.assertEqual(evidence["coverage"], 1)

    def test_recompression_and_scaling_require_structural_verification(self):
        original = self.descriptor("r0")
        for suffix in ("jpeg", "scale"):
            with self.subTest(suffix=suffix):
                result = self.cache.verify(original, self.descriptor(suffix), self.policy)
                self.assertEqual(result["level"], "verified_visual", result)

    def test_same_histogram_is_not_identity(self):
        original = self.descriptor("r0")
        shuffled = self.cache.describe(self.paths["negative-0-shuffled"])
        self.assertTrue(np.allclose(original.histogram, shuffled.histogram))
        result = self.cache.verify(original, shuffled, self.policy)
        self.assertEqual(result["level"], "rejected")

    def test_crops_color_changes_and_arbitrary_rotations_remain_candidates(self):
        for suffix in ("crop", "brightness", "contrast", "rotation_padding"):
            with self.subTest(suffix=suffix):
                result = self.cache.verify(
                    self.descriptor("r0"), self.descriptor(suffix), self.policy
                )
                self.assertIn(result["level"], {"candidate", "rejected"}, result)

    def test_alpha_changes_are_not_ignored(self):
        image = cv2.imread(self.paths["leaf-0-r0"])
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        left = np.dstack((image, alpha))
        right = left.copy()
        right[:30, :30, 3] = 0
        for name, value in (("left", left), ("right", right)):
            cv2.imwrite(str(self.root / f"{name}.png"), value)
        result = self.cache.verify(
            self.cache.describe(str(self.root / "left.png")),
            self.cache.describe(str(self.root / "right.png")),
            self.policy,
        )
        self.assertEqual(result["level"], "candidate")

    def test_invalid_files_are_reported_without_zero_vector_neighbors(self):
        bad = self.root / "bad.jpg"
        bad.write_bytes(b"not an image")
        _, pairs, stats = analyze_images(
            {"bad": str(bad), "good": self.paths["leaf-0-r0"]}, self.policy, self.cache
        )
        self.assertEqual(pairs, [])
        self.assertIn("bad", stats["invalid"])

    def test_exact_family_survives_one_neighbor_limit(self):
        descriptors = {
            key: self.cache.describe(path)
            for key, path in self.paths.items()
            if "mirror" in key or key.rsplit("-", 1)[-1] in {"r0", "r90", "r180", "r270"}
        }
        pairs, _ = candidate_pairs(descriptors, DeduplicationPolicy(candidate_neighbors=1))
        members = {key for pair in pairs for key in pair}
        self.assertEqual(members, set(descriptors))

    def test_cache_reuses_descriptors_and_pairs_and_detects_changed_content(self):
        original = self.descriptor("r0")
        rotated = self.descriptor("r90")
        first = self.cache.verify(original, rotated, self.policy)
        fresh = DescriptorCache(self.root / "cache")
        again = fresh.describe(self.paths["leaf-0-r0"])
        self.assertEqual(fresh.hits, 1)
        self.assertEqual(fresh.verify(again, rotated, self.policy), first)
        self.assertEqual(fresh.pair_hits, 1)
        fresh.verify(again, rotated, DeduplicationPolicy(verification_max_error=0.02))
        self.assertEqual(fresh.pair_hits, 1)
        image = cv2.imread(self.paths["leaf-0-r0"])
        image[0, 0] = 0
        cv2.imwrite(self.paths["leaf-0-r0"], image)
        self.assertNotEqual(fresh.describe(self.paths["leaf-0-r0"]).asset, original.asset)

    def test_uniform_background_is_not_rotation_padding(self):
        image = np.full((120, 160), 255, dtype=np.uint8)
        cv2.ellipse(image, (80, 60), (35, 45), 0, 0, 360, 100, -1)
        self.assertEqual(corner_padding(image), 0)

    def test_corrupted_descriptor_and_pair_cache_are_recomputed(self):
        original = self.descriptor("r0")
        rotated = self.descriptor("r90")
        target = Path(original.thumbnail_path)
        cached = json.loads(target.read_text())
        cached["payload"]["pixels"]["r0"] = "0" * 64
        target.write_text(json.dumps(cached))
        fresh = DescriptorCache(self.root / "cache")
        recovered = fresh.describe(self.paths["leaf-0-r0"])
        self.assertEqual(recovered.pixels, original.pixels)
        self.assertEqual(fresh.hits, 0)
        fresh.verify(recovered, rotated, self.policy)
        pair_file = next((self.root / "cache" / recovered.version / "pairs").glob("*.json"))
        cached = json.loads(pair_file.read_text())
        cached["payload"]["evidence"]["level"] = "rejected"
        pair_file.write_text(json.dumps(cached))
        self.assertEqual(fresh.verify(recovered, rotated, self.policy)["level"], "exact_pixels")
        self.assertEqual(fresh.pair_hits, 0)

    def test_crop_records_asymmetric_coverage_and_unmatched_content(self):
        result = self.cache.verify(self.descriptor("r0"), self.descriptor("crop"), self.policy)
        self.assertEqual(result["level"], "candidate")
        self.assertGreaterEqual(result["inliers"], 8)
        self.assertLess(result["coverage_left"], 0.7)
        self.assertGreater(result["coverage_right"], 0.95)
        self.assertGreater(result["unmatched_content_left"], 0.1)
        self.assertLess(result["reprojection_rmse"], 2.0)

    def test_index_is_bounded_even_when_all_histograms_share_buckets(self):
        original = self.descriptor("r0")
        descriptors = {
            str(i): replace(
                original, asset=f"{i:064x}", pixels={key: f"{i:064x}" for key in original.pixels}
            )
            for i in range(1200)
        }
        # Ensure an exact family remains connected even when it lies outside query windows.
        descriptors["1199"] = replace(descriptors["1199"], pixels=descriptors["0"].pixels)
        metrics = {}
        policy = DeduplicationPolicy(candidate_pool=32, candidate_neighbors=1)
        pairs, _ = candidate_pairs(descriptors, policy, metrics)
        self.assertIn(("0", "1199"), pairs)
        self.assertLessEqual(metrics["scored_neighbors"], 1200 * 32)
        self.assertGreater(metrics["pool_truncated_images"], 0)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(left < right for left, right in pairs))
        reversed_pairs, _ = candidate_pairs(dict(reversed(list(descriptors.items()))), policy)
        self.assertEqual(pairs, reversed_pairs)

    def test_benchmark_reports_index_recall_and_does_not_certify_synthetic_results(self):
        manifest = self.root / "fixture" / "manifest.json"
        report = evaluate_manifest(
            manifest, self.policy, self.root / "cache", compare_exhaustive=True
        )
        self.assertEqual(report["false_positive_pairs"], 0)
        self.assertEqual(report["retrieval_comparison"]["confirmed_pair_recall"], 1)
        self.assertFalse(report["activation"]["ready"])
        self.assertLess(report["family_consistency"]["wilson_95_interval"][0], 0.995)
        self.assertIn("unknown", report["by_source"])

    def test_evaluation_rejects_overlap_with_calibration_even_if_ids_changed(self):
        manifest = self.root / "fixture" / "manifest.json"
        calibration = self.root / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "samples": [
                        {
                            "id": "different",
                            "family": "other-name",
                            "filepath": self.paths["leaf-0-r90"],
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "comparten"):
            evaluate_manifest(
                manifest, self.policy, self.root / "cache", calibration_manifest=calibration
            )

    def test_benchmark_rejects_missing_images_and_malformed_labels(self):
        manifest = self.root / "bad-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "synthetic": False,
                    "samples": [{"id": "a", "family": "f", "filepath": "missing.png"}],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "incompleto"):
            evaluate_manifest(manifest, self.policy, self.root / "cache")
        manifest.write_text(
            json.dumps(
                {
                    "synthetic": False,
                    "samples": [{"id": "a", "family": "", "filepath": "missing.png"}],
                }
            )
        )
        with self.assertRaises(ValueError):
            read_manifest(manifest)

    def test_fresh_split_audit_catches_missing_family_edges(self):
        paths = {"a": self.paths["leaf-0-r0"], "b": self.paths["leaf-0-r90"]}
        with self.assertRaisesRegex(RuntimeError, "Fuga visual"):
            audit_visual_splits(paths, {"a": "train", "b": "test"}, self.policy, self.cache)
        report = audit_visual_splits(paths, {"a": "train", "b": "train"}, self.policy, self.cache)
        self.assertEqual(report["confirmed_cross_split_pairs"], 0)
        report = audit_visual_splits(
            paths, {"a": "train", "b": "test"}, self.policy, self.cache, {frozenset(("a", "b"))}
        )
        self.assertEqual(report["cross_split_pairs_checked"], 0)

    def test_holdout_rejects_same_global_family_across_sources(self):
        manifest = self.root / "fixture" / "manifest.json"
        calibration = self.root / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "samples": [
                        {
                            "id": "other",
                            "family": "leaf-0",
                            "source": "another-source",
                            "filepath": self.paths["leaf-0-brightness"],
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "comparten"):
            evaluate_manifest(
                manifest, self.policy, self.root / "cache", calibration_manifest=calibration
            )


class FamilyTests(TestCase):
    def sample(self, key, **fields):
        return SimpleNamespace(
            id=key, filepath=f"/fixture/{key}.png", source_dataset="source", **fields
        )

    def test_all_constraints_are_unioned_through_excluded_bridges(self):
        a = self.sample("a", capture_group="capture", duplicate_links=[])
        b = self.sample(
            "b",
            capture_group="capture",
            duplicate_links=[{"other_id": "c", "level": "exact_pixels"}],
            status="removed",
        )
        c = self.sample("c", duplicate_links=[])
        groups = relation_groups([a, b, c], captures=True)
        self.assertEqual(groups["a"], groups["c"])
        with self.assertRaisesRegex(RuntimeError, "Fuga"):
            audit_assignments([a, b, c], {"a": "train", "c": "test"})

    def test_candidate_relations_do_not_claim_confirmed_identity(self):
        a = self.sample("a", duplicate_links=[{"other_id": "b", "level": "candidate"}])
        b = self.sample("b", duplicate_links=[])
        confirmed = relation_groups([a, b], confirmed_only=True)
        conservative = relation_groups([a, b])
        self.assertNotEqual(confirmed["a"], confirmed["b"])
        self.assertEqual(conservative["a"], conservative["b"])

    def test_capture_ids_are_scoped_and_location_is_opt_in(self):
        a = self.sample("a", capture_group="1", location="farm")
        b = self.sample("b", capture_group="2", location="farm")
        c = self.sample("c", capture_group="1")
        c.source_dataset = "another-source"
        groups = relation_groups([a, b, c], captures=True)
        self.assertEqual(len(set(groups.values())), 3)
        locations = relation_groups([a, b, c], captures=True, locations=True)
        self.assertEqual(locations["a"], locations["b"])

    def test_family_identity_survives_database_ids_and_input_order(self):
        a = self.sample("a", duplicate_links=[{"other_id": "b", "level": "exact_pixels"}])
        b = self.sample("b", duplicate_links=[])
        before = relation_groups([a, b])
        a.id, b.id = "new-a", "new-b"
        a.duplicate_links[0]["other_id"] = "new-b"
        after = relation_groups([b, a])
        self.assertEqual(before["a"], after["new-a"])
