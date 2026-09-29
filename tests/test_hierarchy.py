from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from agrivision_khaos.ontology import (
    infer_coarse_label,
    load_hierarchical_ontology,
    load_ontology_mapping,
)


class HierarchyOntologyTests(TestCase):
    def test_infer_coarse_label_detects_healthy_and_disease(self):
        # Health terms -> 'sano'
        self.assertEqual(infer_coarse_label("healthy"), "sano")
        self.assertEqual(infer_coarse_label("sano"), "sano")
        self.assertEqual(infer_coarse_label("saglam"), "sano")
        self.assertEqual(infer_coarse_label("olive_healthy_leaf"), "sano")
        self.assertEqual(infer_coarse_label("normal_leaf"), "sano")

        # Disease terms -> 'enfermo'
        self.assertEqual(infer_coarse_label("repilo"), "enfermo")
        self.assertEqual(infer_coarse_label("antracnosis"), "enfermo")
        self.assertEqual(infer_coarse_label("tuberculosis"), "enfermo")
        self.assertEqual(infer_coarse_label("escudete"), "enfermo")
        self.assertEqual(infer_coarse_label("olive_peacock_spot"), "enfermo")
        self.assertEqual(infer_coarse_label("leaf_curl"), "enfermo")
        self.assertEqual(infer_coarse_label("powdery_mildew"), "enfermo")

    def test_infer_coarse_label_contextual(self):
        # In a dataset with healthy peers, an unknown class is assumed abnormal/diseased
        self.assertEqual(
            infer_coarse_label("chlorosis", known_classes=["sano", "chlorosis"]),
            "enfermo",
        )
        # In a dataset of fruit varieties without health terms, retains class name
        self.assertEqual(
            infer_coarse_label("marcona", known_classes=["marcona", "guara"]),
            "marcona",
        )

    def test_flat_ontology_format_loads_and_derives_coarse(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "flat_ontology.yaml"
            yaml_path.write_text(
                "repilo:\n"
                "  - 'Olive Repilo'\n"
                "  - 'Spilocaea'\n"
                "antracnosis:\n"
                "  - 'Anthracnose'\n"
                "sano:\n"
                "  - 'Healthy'\n",
                encoding="utf-8",
            )
            fine_map, coarse_map = load_hierarchical_ontology(yaml_path)
            self.assertEqual(fine_map["Olive Repilo"], "repilo")
            self.assertEqual(fine_map["Spilocaea"], "repilo")
            self.assertEqual(fine_map["Anthracnose"], "antracnosis")
            self.assertEqual(fine_map["Healthy"], "sano")

            # Coarse map automatically derived
            self.assertEqual(coarse_map["repilo"], "enfermo")
            self.assertEqual(coarse_map["antracnosis"], "enfermo")
            self.assertEqual(coarse_map["sano"], "sano")

            # Backward compatibility loader
            legacy_map = load_ontology_mapping(yaml_path)
            self.assertEqual(legacy_map, fine_map)

    def test_hierarchical_ontology_format_explicit_coarse(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "hierarchical_ontology.yaml"
            yaml_path.write_text(
                "classes:\n"
                "  repilo:\n"
                "    - 'Olive Repilo'\n"
                "  sano:\n"
                "    - 'Healthy'\n"
                "coarse_mapping:\n"
                "  diseased:\n"
                "    - repilo\n"
                "  healthy:\n"
                "    - sano\n",
                encoding="utf-8",
            )
            fine_map, coarse_map = load_hierarchical_ontology(yaml_path)
            self.assertEqual(fine_map["Olive Repilo"], "repilo")
            self.assertEqual(fine_map["Healthy"], "sano")
            self.assertEqual(coarse_map["repilo"], "diseased")
            self.assertEqual(coarse_map["sano"], "healthy")

