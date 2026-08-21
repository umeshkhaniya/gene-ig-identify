"""Tests for config-driven class handling in the training workflow."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gene_ig_identify_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "gene_ig_identify_cache"))

from gene_ig_identify.config import load_config
from gene_ig_identify.constants import EXPECTED_EDGE_FEATURES, EXPECTED_NODE_FEATURES
from gene_ig_identify.io.artifacts import load_json
from gene_ig_identify.labels import LABEL_MAPPING
from gene_ig_identify.workflows import train


MODEL_PARAMS = {
    "hidden_dim": 4,
    "dropout": 0.1,
    "num_layers": 2,
}

EXPECTED_EXP00_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
    "CD19",
]

EXPECTED_EXP01_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
    "CD19",
    "IgE",
    "IgFN3-like",
    "SOD",
]

EXPECTED_EXP02_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
]


def graph_with_label(label: int):
    return SimpleNamespace(
        x=torch.zeros((1, EXPECTED_NODE_FEATURES), dtype=torch.float),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, EXPECTED_EDGE_FEATURES), dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        unique_name_file=f"graph_{label}",
    )


class TrainingLabelSpaceTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def load_config_file(self, filename: str):
        return load_config(self.project_root / "config" / "experiments" / filename)

    def assert_model_label_metadata(self, filename: str, expected_experiment: str, expected_name: str, expected_labels: list[str]):
        config = self.load_config_file(filename)
        label_space = train._label_space_from_config(config)
        metadata = train._model_label_metadata(config, label_space)
        expected_mapping = {
            label: index for index, label in enumerate(expected_labels)
        }

        self.assertEqual(metadata["experiment"], expected_experiment)
        self.assertEqual(metadata["name"], expected_name)
        self.assertEqual(metadata["num_classes"], len(expected_labels))
        self.assertEqual(metadata["labels"], expected_labels)
        self.assertEqual(metadata["label_mapping"], expected_mapping)

    def test_exp00_training_label_space_resolves_to_eight_classes(self):
        label_space = train._label_space_from_config(self.load_config_file("exp00_8class.yaml"))

        self.assertEqual(label_space.num_classes, 8)
        self.assertEqual(label_space.label_mapping, LABEL_MAPPING)
        self.assertEqual(label_space.reverse_label_mapping[7], "CD19")

    def test_exp01_training_label_space_resolves_to_eleven_classes(self):
        label_space = train._label_space_from_config(self.load_config_file("exp01_11class.yaml"))

        self.assertEqual(label_space.num_classes, 11)
        self.assertEqual(label_space.label_mapping["IgE"], 8)
        self.assertEqual(label_space.label_mapping["IgFN3-like"], 9)
        self.assertEqual(label_space.label_mapping["SOD"], 10)
        self.assertNotIn("ORF", label_space.label_mapping)
        self.assertNotIn("ORF", label_space.reverse_label_mapping.values())

    def test_exp02_training_label_space_resolves_to_seven_classes(self):
        label_space = train._label_space_from_config(self.load_config_file("exp02_7class.yaml"))

        self.assertEqual(label_space.num_classes, 7)
        self.assertNotIn("CD19", label_space.label_mapping)
        self.assertEqual(label_space.reverse_label_mapping[6], "Lamin")

    def test_make_model_uses_configured_num_classes(self):
        for filename, expected_classes in (
            ("exp00_8class.yaml", 8),
            ("exp01_11class.yaml", 11),
            ("exp02_7class.yaml", 7),
        ):
            with self.subTest(filename=filename):
                label_space = train._label_space_from_config(self.load_config_file(filename))
                model = train._make_model(MODEL_PARAMS, torch.device("cpu"), label_space.num_classes)

                self.assertEqual(label_space.num_classes, expected_classes)
                self.assertEqual(model.lin2.out_features, expected_classes)

    def test_class_weights_use_configured_num_classes(self):
        label_space = train._label_space_from_config(self.load_config_file("exp01_11class.yaml"))

        weights = train._class_weights(
            [graph_with_label(0), graph_with_label(8)],
            torch.device("cpu"),
            label_space.num_classes,
        )

        self.assertEqual(tuple(weights.shape), (11,))
        self.assertGreater(float(weights[0]), 0.0)
        self.assertGreater(float(weights[8]), 0.0)
        self.assertEqual(float(weights[1]), 0.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_exp00_labeled_graph_validation_accepts_highest_exp00_class_id(self):
        label_space = train._label_space_from_config(self.load_config_file("exp00_8class.yaml"))

        train._validate_labeled_graphs(
            [graph_with_label(7)],
            Path("graphs.pt"),
            label_space.num_classes,
        )

    def test_exp02_labeled_graph_validation_rejects_cd19_class_id(self):
        label_space = train._label_space_from_config(self.load_config_file("exp02_7class.yaml"))

        with self.assertRaisesRegex(ValueError, "configured 7-class mapping"):
            train._validate_labeled_graphs(
                [graph_with_label(7)],
                Path("graphs.pt"),
                label_space.num_classes,
            )

    def test_missing_class_warning_uses_configured_reverse_mapping(self):
        label_space = train._label_space_from_config(self.load_config_file("exp01_11class.yaml"))

        with self.assertLogs(train.LOGGER, level="WARNING") as captured:
            train._warn_missing_classes([0, 8], label_space.reverse_label_mapping)

        message = "\n".join(captured.output)
        self.assertIn("2/11 configured labels", message)
        self.assertIn("IgV, IgE", message)
        self.assertIn("SOD", message)
        self.assertNotIn("ORF", message)

    def test_exp00_model_metadata_contains_eight_ordered_labels(self):
        self.assert_model_label_metadata(
            "exp00_8class.yaml",
            "EXP00",
            "8class_baseline",
            EXPECTED_EXP00_LABELS,
        )

    def test_exp01_model_metadata_contains_eleven_ordered_labels(self):
        self.assert_model_label_metadata(
            "exp01_11class.yaml",
            "EXP01",
            "11class_expanded",
            EXPECTED_EXP01_LABELS,
        )

    def test_exp01_model_metadata_contains_new_labels(self):
        config = self.load_config_file("exp01_11class.yaml")
        label_space = train._label_space_from_config(config)
        metadata = train._model_label_metadata(config, label_space)

        for label in ("IgE", "IgFN3-like", "SOD"):
            self.assertIn(label, metadata["labels"])
            self.assertIn(label, metadata["label_mapping"])

        self.assertNotIn("ORF", metadata["labels"])
        self.assertNotIn("ORF", metadata["label_mapping"])

    def test_exp02_model_metadata_contains_seven_labels_without_cd19(self):
        self.assert_model_label_metadata(
            "exp02_7class.yaml",
            "EXP02",
            "7class_no_CD19",
            EXPECTED_EXP02_LABELS,
        )
        config = self.load_config_file("exp02_7class.yaml")
        label_space = train._label_space_from_config(config)
        metadata = train._model_label_metadata(config, label_space)

        self.assertNotIn("CD19", metadata["labels"])
        self.assertNotIn("CD19", metadata["label_mapping"])
        self.assertNotIn("IgE", metadata["labels"])
        self.assertNotIn("IgE", metadata["label_mapping"])
        self.assertNotIn("IgFN3-like", metadata["labels"])
        self.assertNotIn("IgFN3-like", metadata["label_mapping"])
        self.assertNotIn("SOD", metadata["labels"])
        self.assertNotIn("SOD", metadata["label_mapping"])
        self.assertNotIn("ORF", metadata["labels"])
        self.assertNotIn("ORF", metadata["label_mapping"])

    def test_existing_old_exp00_model_config_remains_readable(self):
        model_config = load_json(self.project_root / "results" / "models" / "model_config.json")

        self.assertEqual(model_config["num_classes"], 8)
        self.assertEqual(model_config["node_features"], EXPECTED_NODE_FEATURES)
        self.assertEqual(model_config["edge_in_channels_features"], EXPECTED_EDGE_FEATURES)
        self.assertIn("best_params", model_config)


if __name__ == "__main__":
    unittest.main()
