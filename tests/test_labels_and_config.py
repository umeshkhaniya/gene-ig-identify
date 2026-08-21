"""Tests for the current EXP00 labels and config loading."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from gene_ig_identify import labels
from gene_ig_identify.config import load_config


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

CONFIG_ENV_VARS = [
    "GENE_IG_IDENTIFY_CONFIG",
    "GENE_IG_IDENTIFY_INPUT_DIR",
    "GENE_IG_IDENTIFY_OUTPUT_DIR",
    "GENE_IG_IDENTIFY_ARTIFACTS_DIR",
    "GENE_IG_IDENTIFY_MODELS_DIR",
    "GENE_IG_IDENTIFY_ESM_CACHE",
    "GENE_IG_IDENTIFY_DEVICE",
    "GENE_IG_IDENTIFY_NODE",
]


@contextmanager
def clean_config_environment():
    with patch.dict(os.environ, {}, clear=False):
        for key in CONFIG_ENV_VARS:
            os.environ.pop(key, None)
        yield


class Exp00LabelMappingTests(unittest.TestCase):
    def test_default_exp00_label_mapping_is_stable(self):
        expected_mapping = {
            label: index for index, label in enumerate(EXPECTED_EXP00_LABELS)
        }
        expected_reverse_mapping = {
            index: label for label, index in expected_mapping.items()
        }

        self.assertEqual(labels.DEFAULT_LABELS, EXPECTED_EXP00_LABELS)
        self.assertEqual(labels.LABEL_MAPPING, expected_mapping)
        self.assertEqual(labels.REVERSE_LABEL_MAPPING, expected_reverse_mapping)

    def test_label_mapping_helpers_preserve_input_order(self):
        label_names = ["first", "second", "third"]
        label_mapping = labels.build_label_mapping(label_names)

        self.assertEqual(label_mapping, {"first": 0, "second": 1, "third": 2})
        self.assertEqual(
            labels.build_reverse_label_mapping(label_mapping),
            {0: "first", 1: "second", 2: "third"},
        )


class ConfigDrivenLabelMappingTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def _load_experiment_config(self, filename: str):
        with clean_config_environment():
            return load_config(self.project_root / "config" / "experiments" / filename)

    def _assert_configured_mapping(self, config, expected_labels):
        expected_mapping = {
            label: index for index, label in enumerate(expected_labels)
        }
        expected_reverse_mapping = {
            index: label for label, index in expected_mapping.items()
        }

        self.assertEqual(labels.labels_from_config(config), expected_labels)
        self.assertEqual(labels.label_mapping_from_config(config), expected_mapping)
        self.assertEqual(
            labels.reverse_label_mapping_from_config(config),
            expected_reverse_mapping,
        )

    def test_default_config_is_exp00_with_stable_mapping(self):
        with clean_config_environment():
            config = load_config(self.project_root / "config" / "default.yaml")

        self.assertEqual(config.raw["experiment"]["id"], "EXP00")
        self.assertEqual(config.raw["experiment"]["name"], "8class_baseline")
        self._assert_configured_mapping(config, EXPECTED_EXP00_LABELS)

    def test_exp00_config_produces_stable_eight_class_mapping(self):
        config = self._load_experiment_config("exp00_8class.yaml")

        self.assertEqual(config.raw["experiment"]["id"], "EXP00")
        self._assert_configured_mapping(config, EXPECTED_EXP00_LABELS)

    def test_exp01_config_produces_expected_eleven_class_mapping(self):
        config = self._load_experiment_config("exp01_11class.yaml")

        self.assertEqual(config.raw["experiment"]["id"], "EXP01")
        configured_labels = labels.labels_from_config(config)
        self.assertEqual(len(configured_labels), 11)
        self.assertIn("IgE", configured_labels)
        self.assertIn("IgFN3-like", configured_labels)
        self.assertIn("SOD", configured_labels)
        self.assertNotIn("ORF", configured_labels)
        self._assert_configured_mapping(config, EXPECTED_EXP01_LABELS)

    def test_exp02_config_produces_expected_seven_class_mapping_without_cd19(self):
        config = self._load_experiment_config("exp02_7class.yaml")

        self.assertEqual(config.raw["experiment"]["id"], "EXP02")
        configured_labels = labels.labels_from_config(config)
        self.assertEqual(len(configured_labels), 7)
        self.assertNotIn("CD19", configured_labels)
        self.assertNotIn("IgE", configured_labels)
        self.assertNotIn("IgFN3-like", configured_labels)
        self.assertNotIn("SOD", configured_labels)
        self.assertNotIn("ORF", configured_labels)
        self._assert_configured_mapping(config, EXPECTED_EXP02_LABELS)


class ConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.default_config = self.project_root / "config" / "default.yaml"

    def test_default_config_loads_expected_runtime_paths_and_model_values(self):
        with clean_config_environment():
            config = load_config(self.default_config)

        self.assertEqual(config.project_root, self.project_root)
        self.assertEqual(config.config_path, self.default_config)
        self.assertEqual(config.raw["experiment"]["id"], "EXP00")
        self.assertEqual(config.raw["experiment"]["name"], "8class_baseline")
        self.assertEqual(config.raw["labels"], EXPECTED_EXP00_LABELS)
        self.assertEqual(config.runtime["target"], "biowulf")
        self.assertEqual(config.runtime["device"], "auto")
        self.assertEqual(config.paths["input_dir"], "input")
        self.assertEqual(config.paths["models_dir"], "results/models")
        self.assertEqual(config.executables["node"], "node")
        self.assertEqual(config.model["node_feature_dim"], 1320)
        self.assertEqual(config.model["edge_feature_dim"], 8)
        self.assertEqual(config.model["default_confidence_threshold"], 0.5)

    def test_exp00_experiment_config_merges_with_default_config(self):
        exp00_config = self.project_root / "config" / "experiments" / "exp00_8class.yaml"

        with clean_config_environment():
            config = load_config(exp00_config)

        self.assertEqual(config.config_path, exp00_config)
        self.assertEqual(config.raw["experiment"]["id"], "EXP00")
        self.assertEqual(config.raw["experiment"]["name"], "8class_baseline")
        self.assertEqual(config.raw["labels"], EXPECTED_EXP00_LABELS)
        self.assertEqual(config.runtime["target"], "biowulf")
        self.assertEqual(config.paths["models_dir"], "results/models")
        self.assertEqual(config.model["node_feature_dim"], 1320)

    def test_environment_overrides_loaded_config_values(self):
        with clean_config_environment():
            with patch.dict(
                os.environ,
                {
                    "GENE_IG_IDENTIFY_OUTPUT_DIR": "custom_output",
                    "GENE_IG_IDENTIFY_DEVICE": "cpu",
                },
            ):
                config = load_config(self.default_config)

        self.assertEqual(config.paths["output_dir"], "custom_output")
        self.assertEqual(config.runtime["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
