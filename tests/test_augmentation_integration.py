from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import cv2
import fiftyone as fo
import numpy as np

from agrivision_khaos.augmentation_benchmark import generate_fixture
from agrivision_khaos.deduplication import (
    apply_stored_decisions,
    detect_augmentation_duplicates,
    detect_exact_duplicates,
    process_duplicates,
    review_pairs,
)
from agrivision_khaos.models import CurationPolicy, DeduplicationPolicy
from agrivision_khaos.pipeline import (
    Decision,
    apply_second_opinion,
    audit_duplicate_representatives,
    decision_for_tagged_duplicates,
    load_policy,
    partition_dataset,
    prepare_duplicate_run,
    run_duplicate_phases,
    write_decisions,
)


@unittest.skipUnless(
    os.environ.get("AGRIVISION_RUN_INTEGRATION") == "1", "Requiere MongoDB temporal"
)
class AugmentationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = generate_fixture(self.root / "fixture", families=1)
        self.paths = {
            row["id"]: str(manifest.parent / row["filepath"])
            for row in json.loads(manifest.read_text())["samples"]
        }
        self.dataset = fo.Dataset(f"augmentation-test-{uuid4().hex}")
        self.addCleanup(lambda: self.dataset.delete())

    def add(self, key, *, sharpness=10, boxes=None):
        sample = fo.Sample(filepath=self.paths[key])
        sample["curation"] = fo.DynamicEmbeddedDocument(
            status="kept", phase="quality", reason="good", confidence=1.0
        )
        sample["normalized_labels"] = ["leaf"]
        sample["normalized_label"] = "leaf"
        sample["task_type"] = "classification" if boxes is None else "detection"
        sample["annotation_valid"] = True
        sample["augmentation_sharpness"] = sharpness
        sample["source_dataset"] = "fixture"
        if boxes is not None:
            sample["ground_truth_detections"] = fo.Detections(
                detections=[fo.Detection(label="leaf", bounding_box=box) for box in boxes]
            )
        self.dataset.add_sample(sample)
        return sample

    def test_annotation_boxes_are_preserved_despite_identical_bytes_and_class(self):
        self.add("leaf-0-r0", boxes=[[0.1, 0.1, 0.2, 0.2]])
        self.add("leaf-0-r0", boxes=[[0.5, 0.5, 0.2, 0.2]])
        count, pairs = detect_exact_duplicates(self.dataset, "redundant_exact")
        self.assertEqual((count, pairs), (0, []))
        self.assertEqual(len(self.dataset.match_tags("redundant_exact_label_conflict")), 2)

    def test_instance_masks_are_not_discarded_based_on_equal_boxes(self):
        a = self.add("leaf-0-r0", boxes=[[0.1, 0.1, 0.2, 0.2]])
        self.add("leaf-0-r0", boxes=[[0.1, 0.1, 0.2, 0.2]])
        a.ground_truth_detections.detections[0].mask = np.ones((20, 20), dtype=np.uint8)
        a.save()
        count, _ = detect_exact_duplicates(self.dataset, "redundant_exact")
        self.assertEqual(count, 0)

    def test_stored_apply_resolves_duplicate_review_and_preserves_records(self):
        self.add("leaf-0-r0")
        self.add("leaf-0-r0")
        detect_exact_duplicates(self.dataset, "redundant_exact")
        decisions = decision_for_tagged_duplicates(
            self.dataset,
            "redundant_exact",
            "exact_duplicates",
            "Redundante (Exacta)",
            status="review",
        )
        write_decisions(self.dataset, decisions, "exact_duplicates")
        result = apply_stored_decisions(
            self.dataset, "redundant_exact", DeduplicationPolicy(), self.root / "cache"
        )
        self.assertEqual(result.removed, 1)
        self.assertEqual(len(self.dataset), 2)
        audit_duplicate_representatives(self.dataset)

    def test_rotated_detection_boxes_must_follow_image_transform(self):
        self.add("leaf-0-r0", boxes=[[0.1, 0.2, 0.3, 0.4]])
        self.add("leaf-0-r90", boxes=[[0.2, 0.6, 0.4, 0.3]])
        count, _ = detect_augmentation_duplicates(self.dataset, "redundant_augmented", 0.92)
        self.assertEqual(count, 1)
        self.assertEqual(len(self.dataset.match_tags("redundant_augmented_label_conflict")), 0)

    def test_method_relations_survive_and_stale_results_are_cleared(self):
        a, b, c = [self.add(f"leaf-0-{kind}") for kind in ("r0", "r90", "r180")]
        process_duplicates(self.dataset, {a.id: [b.id]}, "redundant_exact")
        process_duplicates(self.dataset, {b.id: [c.id]}, "redundant_semantic")
        self.assertEqual(len(set(self.dataset.values("duplicate_cluster_id"))), 1)
        process_duplicates(self.dataset, {}, "redundant_semantic")
        self.assertEqual(
            self.dataset[a.id].duplicate_cluster_id, self.dataset[b.id].duplicate_cluster_id
        )
        self.assertNotEqual(
            self.dataset[a.id].duplicate_cluster_id, self.dataset[c.id].duplicate_cluster_id
        )
        self.assertEqual(len(self.dataset.match_tags("redundant_semantic")), 0)
        self.assertTrue(all(not sample.redundant_semantic_evidence for sample in self.dataset))

    def test_transitive_visual_neighbors_do_not_authorize_indirect_substitution(self):
        a = self.add("leaf-0-r0", sharpness=30)
        b = self.add("leaf-0-r90", sharpness=20)
        c = self.add("leaf-0-r180", sharpness=10)
        _, pairs = process_duplicates(
            self.dataset, {a.id: [b.id], b.id: [c.id]}, "redundant_semantic"
        )
        self.assertEqual(pairs, [(a.id, b.id)])

    def test_exact_transform_removals_are_repeatable_and_have_kept_representatives(self):
        for suffix in (
            "r0",
            "r90",
            "r180",
            "r270",
            "mirror_r0",
            "mirror_r90",
            "mirror_r180",
            "mirror_r270",
        ):
            self.add(f"leaf-0-{suffix}")
        policy = CurationPolicy(
            deduplication=DeduplicationPolicy(
                exact_enabled=False,
                semantic_enabled=False,
                augmentation_action="remove",
                remove_exact_transforms=True,
            )
        )
        run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        first = {
            sample.id: (sample.curation.status, sample.duplicate_family_id)
            for sample in self.dataset
        }
        self.assertEqual(sum(status == "kept" for status, _ in first.values()), 1)
        self.assertEqual(sum(status == "removed" for status, _ in first.values()), 7)
        audit_duplicate_representatives(self.dataset)
        run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        second = {
            sample.id: (sample.curation.status, sample.duplicate_family_id)
            for sample in self.dataset
        }
        self.assertEqual(first, second)
        self.assertEqual(self.dataset.info["augmentation_analysis"]["descriptor_hits"], 8)

    def test_cpu_profile_removes_exact_transforms_without_semantic_inference(self):
        suffixes = (
            "r0", "r90", "r180", "r270",
            "mirror_r0", "mirror_r90", "mirror_r180", "mirror_r270",
        )
        for suffix in suffixes:
            self.add(f"leaf-0-{suffix}")
        policy = load_policy(Path(__file__).parents[1] / "configs/quality-first.yaml")
        self.assertFalse(policy.quality.ocr_enabled)
        with patch("agrivision_khaos.deduplication.detect_semantic_duplicates") as semantic:
            run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
            semantic.assert_not_called()
        statuses = self.dataset.values("curation.status")
        self.assertEqual(statuses.count("kept"), 1)
        self.assertEqual(statuses.count("removed"), 7)
        self.assertEqual(len(self.dataset), 8)
        for sample in self.dataset:
            self.assertTrue(Path(sample.filepath).is_file())
            height, width = cv2.imread(sample.filepath).shape[:2]
            self.assertEqual(sample.augmentation_width, width)
            self.assertEqual(sample.augmentation_height, height)
        audit_duplicate_representatives(self.dataset)

    def test_remove_policy_cannot_remove_histogram_only_or_lossy_matches(self):
        self.add("leaf-0-r0")
        self.add("leaf-0-jpeg")
        self.add("negative-0-shuffled")
        policy = CurationPolicy(
            deduplication=DeduplicationPolicy(
                exact_enabled=False,
                semantic_enabled=False,
                augmentation_action="remove",
                remove_exact_transforms=True,
            )
        )
        run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        statuses = self.dataset.values("curation.status")
        self.assertNotIn("removed", statuses)
        self.assertEqual(statuses[-1], "kept")
        self.assertIn("review", statuses)

    def test_semantic_proposals_are_verified_before_marking_samples_for_review(self):
        a, b = self.add("leaf-0-r0"), self.add("negative-0-shuffled")
        policy = CurationPolicy(deduplication=DeduplicationPolicy(exact_enabled=False))
        with patch(
            "agrivision_khaos.deduplication.detect_semantic_duplicates",
            side_effect=lambda dataset, tag, threshold: process_duplicates(
                dataset, {a.id: [b.id]}, tag
            ),
        ):
            run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        self.assertEqual(self.dataset.values("curation.status"), ["kept", "kept"])

    def test_heuristic_confidence_does_not_bypass_drop_limits(self):
        samples = [self.add("leaf-0-r0") for _ in range(10)]
        decisions = {
            sample.id: Decision("removed", "augmentation_duplicates", "heuristic", 0.99)
            for sample in samples[:9]
        }
        apply_second_opinion(samples, decisions, 0.4, 0.65)
        self.assertTrue(all(decision.status == "review" for decision in decisions.values()))

    def test_persisted_apply_checks_current_annotations_before_mutation(self):
        self.add("leaf-0-r0")
        self.add("leaf-0-r0")
        detect_exact_duplicates(self.dataset, "redundant_exact")
        candidate = self.dataset.match_tags("redundant_exact").first()
        candidate["normalized_labels"] = ["different"]
        candidate.save()
        with self.assertRaisesRegex(ValueError, "anotaciones"):
            apply_stored_decisions(
                self.dataset, "redundant_exact", DeduplicationPolicy(), self.root / "cache"
            )
        self.assertEqual(self.dataset.values("curation.status"), ["kept", "kept"])

    def test_manual_rejection_splits_a_family_and_survives_redetection(self):
        a, b = self.add("leaf-0-r0"), self.add("leaf-0-r90")
        detect_augmentation_duplicates(self.dataset, "redundant_augmented", 0.92)
        review_pairs(self.dataset, [{"left": a.id, "right": b.id, "decision": "reject"}])
        self.assertNotEqual(
            self.dataset[a.id].duplicate_family_id, self.dataset[b.id].duplicate_family_id
        )
        detect_augmentation_duplicates(self.dataset, "redundant_augmented", 0.92)
        self.assertEqual(len(self.dataset.match_tags("redundant_augmented")), 0)
        self.assertNotEqual(
            self.dataset[a.id].duplicate_family_id, self.dataset[b.id].duplicate_family_id
        )

    def test_quality_review_reason_survives_duplicate_decision(self):
        self.add("leaf-0-r0")
        b = self.add("leaf-0-r90")
        b.curation = fo.DynamicEmbeddedDocument(
            status="review", phase="quality", reason="blur", confidence=0.7
        )
        b.save()
        detect_augmentation_duplicates(self.dataset, "redundant_augmented", 0.92)
        decisions = decision_for_tagged_duplicates(
            self.dataset,
            "redundant_augmented",
            "augmentation_duplicates",
            "duplicate",
            status="removed",
        )
        write_decisions(self.dataset, decisions, "augmentation_duplicates")
        self.assertEqual(self.dataset[b.id].curation.status, "review")
        self.assertIn("blur", self.dataset[b.id].curation.review_reasons)

    def test_split_constraints_include_removed_capture_bridge(self):
        a, b, c = [self.add(f"leaf-0-{kind}") for kind in ("r0", "r90", "r180")]
        process_duplicates(self.dataset, {b.id: [c.id]}, "redundant_semantic")
        for sample in (self.dataset[a.id], self.dataset[b.id]):
            sample["capture_group"] = "capture-1"
            if sample.id == b.id:
                sample.curation.status = "removed"
            sample.save()
        partition_dataset(self.dataset)
        split_a = set(self.dataset[a.id].tags) & {"train", "val", "test"}
        split_c = set(self.dataset[c.id].tags) & {"train", "val", "test"}
        self.assertEqual(split_a, split_c)
        self.assertEqual(len(split_a), 1)

    def test_lossless_single_pixel_change_invalidates_exact_removal(self):
        self.add("leaf-0-r0")
        other_path = self.root / "copy.png"
        other_path.write_bytes(Path(self.paths["leaf-0-r0"]).read_bytes())
        self.paths["copy"] = str(other_path)
        self.add("copy")
        detect_exact_duplicates(self.dataset, "redundant_exact")
        image = cv2.imread(str(other_path))
        image[0, 0] = 0
        cv2.imwrite(str(other_path), image)
        with self.assertRaises(ValueError):
            apply_stored_decisions(
                self.dataset, "redundant_exact", DeduplicationPolicy(), self.root / "cache"
            )

    def test_standalone_redetection_restores_only_its_own_decisions(self):
        self.add("leaf-0-r0")
        self.add("leaf-0-r90")
        prepare_duplicate_run(self.dataset, method="redundant_augmented")
        detect_augmentation_duplicates(
            self.dataset, "redundant_augmented", 0.92, DeduplicationPolicy(), self.root / "cache"
        )
        decisions = decision_for_tagged_duplicates(
            self.dataset,
            "redundant_augmented",
            "augmentation_duplicates",
            "Duplicado verificado",
            status="review",
        )
        write_decisions(self.dataset, decisions, "augmentation_duplicates")
        first = {sample.id: sample.curation.status for sample in self.dataset}
        self.assertEqual(list(first.values()).count("review"), 1)
        prepare_duplicate_run(self.dataset, method="redundant_augmented")
        detect_augmentation_duplicates(
            self.dataset, "redundant_augmented", 0.92, DeduplicationPolicy(), self.root / "cache"
        )
        decisions = decision_for_tagged_duplicates(
            self.dataset,
            "redundant_augmented",
            "augmentation_duplicates",
            "Duplicado verificado",
            status="review",
        )
        write_decisions(self.dataset, decisions, "augmentation_duplicates")
        self.assertEqual(first, {sample.id: sample.curation.status for sample in self.dataset})
        reviewed = next(sample for sample in self.dataset if sample.curation.status == "review")
        self.assertGreaterEqual(len(reviewed.curation_history), 3)

    def test_isolated_low_information_image_is_reviewed(self):
        image = np.full((192, 256, 3), 128, dtype=np.uint8)
        cv2.imwrite(self.paths["leaf-0-r0"], image)
        sample = self.add("leaf-0-r0")
        policy = CurationPolicy(
            deduplication=DeduplicationPolicy(exact_enabled=False, semantic_enabled=False)
        )
        run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        self.assertEqual(self.dataset[sample.id].curation.status, "review")

    def test_manual_export_validation_does_not_partially_apply_tags(self):
        from agrivision_khaos.export import sync_manual_decisions

        a, b = self.add("leaf-0-r0"), self.add("leaf-0-r90")
        write_decisions(
            self.dataset,
            {sample.id: Decision("review", "quality", "inspect", 0.5) for sample in (a, b)},
            "quality",
        )
        self.dataset.select([a.id]).tag_samples("kept")
        with self.assertRaisesRegex(RuntimeError, "revisión"):
            sync_manual_decisions(self.dataset)
        self.assertEqual(self.dataset[a.id].curation.status, "review")
        self.dataset.select([b.id]).tag_samples("kept")
        sync_manual_decisions(self.dataset)
        self.assertEqual(self.dataset[a.id].curation_history[-1]["action"], "human_review")

    def test_invalid_split_policy_preserves_previous_tags(self):
        sample = self.add("leaf-0-r0")
        self.dataset.tag_samples("test")
        with self.assertRaises(ValueError):
            partition_dataset(self.dataset, train_p=0.9, val_p=0.2, test_p=0.1)
        self.assertIn("test", self.dataset[sample.id].tags)

    def test_cli_detection_repeat_and_reset_are_reversible(self):
        from agrivision_khaos.deduplication import main

        self.add("leaf-0-r0")
        self.add("leaf-0-r90")
        common = [
            "agrivision-deduplicate",
            "--dataset",
            self.dataset.name,
            "--cache-dir",
            str(self.root / "cache"),
            "--lock-dir",
            str(self.root / "locks"),
        ]
        for _ in range(2):
            with patch("sys.argv", common + ["--method", "augmented"]):
                main()
            self.assertEqual(self.dataset.count_values("curation.status"), {"kept": 1, "review": 1})
        with patch("sys.argv", common + ["--reset"]):
            main()
        self.assertEqual(self.dataset.count_values("curation.status"), {"kept": 2})
        self.assertTrue(all(not sample.duplicate_links for sample in self.dataset))

    def test_export_audit_revalidates_removed_assets_after_content_change(self):
        from agrivision_khaos.augmentation import DescriptorCache

        self.add("leaf-0-r0")
        self.add("leaf-0-r90")
        policy = CurationPolicy(
            deduplication=DeduplicationPolicy(
                exact_enabled=False,
                semantic_enabled=False,
                augmentation_action="remove",
                remove_exact_transforms=True,
            )
        )
        run_duplicate_phases(self.dataset, self.root / "work", 0.4, 0.65, policy)
        removed = next(sample for sample in self.dataset if sample.curation.status == "removed")
        audit_duplicate_representatives(self.dataset, DescriptorCache(self.root / "audit"))
        image = cv2.imread(removed.filepath)
        image[0, 0] = 0
        cv2.imwrite(removed.filepath, image)
        with self.assertRaisesRegex(RuntimeError, "obsoleta"):
            audit_duplicate_representatives(self.dataset, DescriptorCache(self.root / "audit"))
