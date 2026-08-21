"""Tests for experiment-isolated artifact paths."""

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gene_ig_identify_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "gene_ig_identify_cache"))

from gene_ig_identify import cli
from gene_ig_identify.config import load_config
from gene_ig_identify.models.inference import load_model_artifacts, load_model_label_space
from gene_ig_identify.paths import (
    get_experiment_dir,
    get_experiment_metrics_dir,
    get_experiment_models_dir,
    get_experiment_predictions_dir,
    get_path,
)
from gene_ig_identify.scripts.evaluate_test_graphs import (
    _default_metrics_dir,
    _default_predictions_dir,
)


class ExperimentPathTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def load_experiment_config(self, filename: str):
        return load_config(self.project_root / "config" / "experiments" / filename)

    def test_experiment_root_paths_are_config_driven(self):
        for filename, experiment_id in (
            ("exp00_8class.yaml", "EXP00"),
            ("exp01_11class.yaml", "EXP01"),
            ("exp02_7class.yaml", "EXP02"),
        ):
            with self.subTest(filename=filename):
                config = self.load_experiment_config(filename)

                self.assertEqual(
                    get_experiment_dir(config),
                    self.project_root / "results" / "experiments" / experiment_id,
                )

    def test_model_output_paths_are_isolated_by_experiment(self):
        model_dirs = {
            experiment_id: get_experiment_models_dir(self.load_experiment_config(filename))
            for filename, experiment_id in (
                ("exp00_8class.yaml", "EXP00"),
                ("exp01_11class.yaml", "EXP01"),
                ("exp02_7class.yaml", "EXP02"),
            )
        }

        self.assertEqual(len(set(model_dirs.values())), 3)
        self.assertEqual(model_dirs["EXP00"], self.project_root / "results" / "experiments" / "EXP00" / "models")
        self.assertEqual(model_dirs["EXP01"], self.project_root / "results" / "experiments" / "EXP01" / "models")
        self.assertEqual(model_dirs["EXP02"], self.project_root / "results" / "experiments" / "EXP02" / "models")

    def test_prediction_output_paths_are_isolated_by_experiment(self):
        prediction_dirs = {
            experiment_id: get_experiment_predictions_dir(self.load_experiment_config(filename))
            for filename, experiment_id in (
                ("exp00_8class.yaml", "EXP00"),
                ("exp01_11class.yaml", "EXP01"),
                ("exp02_7class.yaml", "EXP02"),
            )
        }

        self.assertEqual(len(set(prediction_dirs.values())), 3)
        self.assertEqual(
            prediction_dirs["EXP00"],
            self.project_root / "results" / "experiments" / "EXP00" / "predictions",
        )
        self.assertEqual(
            prediction_dirs["EXP01"],
            self.project_root / "results" / "experiments" / "EXP01" / "predictions",
        )
        self.assertEqual(
            prediction_dirs["EXP02"],
            self.project_root / "results" / "experiments" / "EXP02" / "predictions",
        )

    def test_metrics_output_paths_are_isolated_by_experiment(self):
        config = self.load_experiment_config("exp01_11class.yaml")

        self.assertEqual(
            get_experiment_metrics_dir(config),
            self.project_root / "results" / "experiments" / "EXP01" / "metrics",
        )

    def test_existing_legacy_results_models_path_remains_readable(self):
        config = load_config(self.project_root / "config" / "default.yaml")
        legacy_models_dir = get_path(config, "models_dir")

        self.assertEqual(legacy_models_dir, self.project_root / "results" / "models")
        self.assertTrue((legacy_models_dir / "model_config.json").exists())
        model, _, label_space = load_model_artifacts(legacy_models_dir, torch.device("cpu"))

        self.assertEqual(label_space.num_classes, 8)
        self.assertEqual(model.lin2.out_features, 8)

    def test_official_exp00_baseline_is_available_at_experiment_model_path(self):
        config = self.load_experiment_config("exp00_8class.yaml")
        exp00_models_dir = get_experiment_models_dir(config)

        for artifact_name in ("best_graph_model.pth", "best_hyperparameters.json", "model_config.json"):
            with self.subTest(artifact_name=artifact_name):
                experiment_path = exp00_models_dir / artifact_name

                self.assertTrue(experiment_path.exists())

        self.assertEqual(load_model_label_space(exp00_models_dir).num_classes, 8)

    def test_experiment_ids_cannot_escape_or_alias_output_directories(self):
        config = load_config(
            self.project_root / "config" / "default.yaml",
            overrides={"experiment": {"id": "../EXP01"}},
        )

        with self.assertRaisesRegex(ValueError, "Invalid experiment id"):
            get_experiment_dir(config)

    def test_evaluate_test_graph_defaults_use_experiment_sibling_dirs(self):
        model_dir = self.project_root / "results" / "experiments" / "EXP00" / "models"

        self.assertEqual(_default_predictions_dir(model_dir), model_dir.parent / "predictions")
        self.assertEqual(_default_metrics_dir(model_dir), model_dir.parent / "metrics")

    def test_evaluate_test_graph_defaults_keep_legacy_model_dir(self):
        legacy_model_dir = self.project_root / "results" / "models"

        self.assertEqual(_default_predictions_dir(legacy_model_dir), legacy_model_dir)
        self.assertEqual(_default_metrics_dir(legacy_model_dir), legacy_model_dir)


class ExperimentCliPathTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def test_train_cli_defaults_to_experiment_model_and_metric_dirs(self):
        argv = [
            "gene-ig-identify",
            "--config",
            str(self.project_root / "config" / "experiments" / "exp01_11class.yaml"),
            "train",
            "--graphs-file",
            "graphs.pt",
            "--graph-lookup-file",
            "lookup.pt",
            "--epochs",
            "1",
            "--trials",
            "1",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("gene_ig_identify.workflows.train.run") as train_run,
        ):
            cli.main()

        self.assertEqual(
            train_run.call_args.kwargs["output_dir"],
            self.project_root / "results" / "experiments" / "EXP01" / "models",
        )
        self.assertEqual(
            train_run.call_args.kwargs["metrics_dir"],
            self.project_root / "results" / "experiments" / "EXP01" / "metrics",
        )

    def test_train_cli_keeps_explicit_output_dir_behavior(self):
        argv = [
            "gene-ig-identify",
            "--config",
            str(self.project_root / "config" / "experiments" / "exp00_8class.yaml"),
            "train",
            "--graphs-file",
            "graphs.pt",
            "--graph-lookup-file",
            "lookup.pt",
            "--output-dir",
            "custom_models",
            "--epochs",
            "1",
            "--trials",
            "1",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("gene_ig_identify.workflows.train.run") as train_run,
        ):
            cli.main()

        self.assertEqual(train_run.call_args.kwargs["output_dir"], self.project_root / "custom_models")
        self.assertIsNone(train_run.call_args.kwargs["metrics_dir"])

    def test_predict_cli_defaults_to_experiment_model_and_prediction_dirs(self):
        argv = [
            "gene-ig-identify",
            "--config",
            str(self.project_root / "config" / "experiments" / "exp02_7class.yaml"),
            "predict",
            "dataset",
            "--graphs-file",
            "graphs.pt",
            "--excel-file",
            "domains.csv",
        ]

        with (
            patch.object(sys, "argv", argv),
            patch("gene_ig_identify.workflows.predict.run_excel_predictions") as predict_run,
        ):
            cli.main()

        self.assertEqual(
            predict_run.call_args.kwargs["model_dir"],
            self.project_root / "results" / "experiments" / "EXP02" / "models",
        )
        self.assertEqual(
            predict_run.call_args.kwargs["output_dir"],
            self.project_root / "results" / "experiments" / "EXP02" / "predictions",
        )


if __name__ == "__main__":
    unittest.main()
