"""Tests for model-specific label mappings during inference and prediction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gene_ig_identify.config import load_config
from gene_ig_identify.labels import LABEL_MAPPING, REVERSE_LABEL_MAPPING
from gene_ig_identify.models import inference
from gene_ig_identify.workflows import predict


EXP00_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
    "CD19",
]

EXP01_LABELS = [
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

EXP02_LABELS = [
    "IgV",
    "IgC1",
    "IgC2",
    "IgI",
    "Cadherin",
    "IgFN3",
    "Lamin",
]


def model_config_for(labels: list[str], experiment: str = "EXPTEST", name: str = "test") -> dict:
    return {
        "experiment": experiment,
        "name": name,
        "labels": labels,
        "label_mapping": {label: index for index, label in enumerate(labels)},
        "num_classes": len(labels),
    }


def label_space_for(labels: list[str], experiment: str = "EXPTEST", name: str = "test") -> inference.ModelLabelSpace:
    return inference.model_label_space_from_config(model_config_for(labels, experiment, name))


class ModelLabelSpaceTests(unittest.TestCase):
    def test_exp00_model_metadata_maps_class_ids_correctly(self):
        label_space = label_space_for(EXP00_LABELS, "EXP00", "8class_baseline")

        self.assertEqual(label_space.num_classes, 8)
        self.assertEqual(label_space.reverse_label_mapping, dict(enumerate(EXP00_LABELS)))
        self.assertEqual(label_space.label_mapping["CD19"], 7)

    def test_exp01_new_class_ids_map_correctly(self):
        label_space = label_space_for(EXP01_LABELS, "EXP01", "11class_expanded")

        self.assertEqual(label_space.num_classes, 11)
        self.assertEqual(label_space.reverse_label_mapping[8], "IgE")
        self.assertEqual(label_space.reverse_label_mapping[9], "IgFN3-like")
        self.assertEqual(label_space.reverse_label_mapping[10], "SOD")
        self.assertNotIn("ORF", label_space.label_mapping)
        self.assertNotIn("ORF", label_space.reverse_label_mapping.values())

    def test_exp02_class_ids_map_correctly_without_cd19(self):
        label_space = label_space_for(EXP02_LABELS, "EXP02", "7class_no_CD19")

        self.assertEqual(label_space.num_classes, 7)
        self.assertEqual(label_space.reverse_label_mapping, dict(enumerate(EXP02_LABELS)))
        self.assertNotIn("CD19", label_space.label_mapping)
        self.assertNotIn("CD19", label_space.reverse_label_mapping.values())

    def test_old_exp00_model_config_without_label_metadata_uses_legacy_mapping(self):
        label_space = inference.model_label_space_from_config({"num_classes": 8})

        self.assertTrue(label_space.is_legacy)
        self.assertEqual(label_space.experiment, "EXP00")
        self.assertEqual(label_space.name, "8class_baseline")
        self.assertEqual(label_space.reverse_label_mapping, REVERSE_LABEL_MAPPING)

    def test_model_metadata_takes_precedence_over_global_mapping(self):
        label_space = label_space_for(["CustomZero", "CustomOne"])

        self.assertEqual(REVERSE_LABEL_MAPPING[0], "IgV")
        self.assertEqual(label_space.reverse_label_mapping[0], "CustomZero")
        self.assertEqual(label_space.reverse_label_mapping[1], "CustomOne")

    def test_other_is_not_a_model_label(self):
        self.assertNotIn("Other", LABEL_MAPPING)
        self.assertNotIn("Other", REVERSE_LABEL_MAPPING.values())

        for labels, expected_classes in (
            (EXP00_LABELS, 8),
            (EXP01_LABELS, 11),
            (EXP02_LABELS, 7),
        ):
            with self.subTest(labels=labels):
                label_space = label_space_for(labels)

                self.assertEqual(label_space.num_classes, expected_classes)
                self.assertNotIn("Other", label_space.labels)
                self.assertNotIn("Other", label_space.label_mapping)
                self.assertNotIn("Other", label_space.reverse_label_mapping.values())


class PredictionWorkflowLabelMappingTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config = load_config(self.project_root / "config" / "default.yaml")

    def run_prediction_with_label_space(
        self,
        base_dir: Path,
        label_space: inference.ModelLabelSpace,
        predicted_class_id: int,
        true_label: str | None = None,
        graph_name: str = "1ABC_A_1_1",
        confidence: float = 0.95,
    ) -> dict[str, pd.DataFrame]:
        model_dir = base_dir / "model"
        model_dir.mkdir()
        for artifact_name in ("best_graph_model.pth", "best_hyperparameters.json", "model_config.json"):
            (model_dir / artifact_name).write_text("placeholder", encoding="utf-8")

        graphs_file = base_dir / "graphs.pt"
        graphs_file.write_text("placeholder", encoding="utf-8")
        output_dir = base_dir / "output"
        input_table = base_dir / "domains.csv"
        table_data = {
            "pdbid_chain": ["1ABC_A"],
            "igdomain_res_range": ["1_1"],
        }
        if true_label is not None:
            table_data["ig_type"] = [true_label]
        pd.DataFrame(table_data).to_csv(input_table, index=False)

        remaining_probability = (1.0 - confidence) / max(label_space.num_classes - 1, 1)
        probabilities = np.full((1, label_space.num_classes), remaining_probability, dtype=float)
        probabilities[0, predicted_class_id] = confidence
        fake_graphs = [
            SimpleNamespace(
                unique_name_file=graph_name,
                source_row_index=0,
                x=object(),
                edge_index=object(),
                edge_attr=object(),
            )
        ]
        written: dict[str, pd.DataFrame] = {}

        def fake_write_excel(df: pd.DataFrame, path: str | Path) -> Path:
            output_path = Path(path)
            written[output_path.name] = df.copy()
            return output_path

        with (
            patch.object(predict, "load_torch", return_value=fake_graphs),
            patch.object(predict, "write_excel", side_effect=fake_write_excel),
            patch("gene_ig_identify.models.inference.resolve_device", return_value=torch.device("cpu")),
            patch(
                "gene_ig_identify.models.inference.load_model_artifacts",
                return_value=(object(), {"batch_size": 1}, label_space),
            ),
            patch(
                "gene_ig_identify.models.inference.predict_graphs",
                return_value=(
                    np.array([predicted_class_id]),
                    probabilities,
                    np.array([]),
                    [graph_name],
                ),
            ),
        ):
            predict.run_excel_predictions(
                self.config,
                graphs_file=graphs_file,
                excel_file=input_table,
                model_dir=model_dir,
                output_dir=output_dir,
            )

        return written

    def test_exp02_prediction_allows_cd19_true_label_without_special_handling(self):
        label_space = label_space_for(EXP02_LABELS, "EXP02", "7class_no_CD19")

        with TemporaryDirectory() as tmp_dir:
            written = self.run_prediction_with_label_space(
                Path(tmp_dir),
                label_space,
                predicted_class_id=5,
                true_label="CD19",
            )

        result_df = written["domains_with_predictions.xlsx"]
        detail_df = written["domains_prediction_details.xlsx"]
        self.assertEqual(result_df.loc[0, "ig_type"], "CD19")
        self.assertEqual(result_df.loc[0, "predicted_class_id"], 5)
        self.assertEqual(result_df.loc[0, "predicted_label"], "IgFN3")
        self.assertIn(result_df.loc[0, "predicted_label"], EXP02_LABELS)
        self.assertEqual(detail_df.loc[0, "true_label"], "CD19")

    def test_prediction_uses_model_metadata_not_global_mapping(self):
        label_space = label_space_for(EXP01_LABELS, "EXP01", "11class_expanded")

        with TemporaryDirectory() as tmp_dir:
            written = self.run_prediction_with_label_space(
                Path(tmp_dir),
                label_space,
                predicted_class_id=8,
            )

        result_df = written["domains_with_predictions.xlsx"]
        self.assertNotIn(8, REVERSE_LABEL_MAPPING)
        self.assertEqual(result_df.loc[0, "predicted_class_id"], 8)
        self.assertEqual(result_df.loc[0, "predicted_label"], "IgE")

    def test_prediction_below_confidence_threshold_becomes_other(self):
        label_space = label_space_for(EXP00_LABELS, "EXP00", "8class_baseline")

        with TemporaryDirectory() as tmp_dir:
            written = self.run_prediction_with_label_space(
                Path(tmp_dir),
                label_space,
                predicted_class_id=0,
                confidence=0.49,
            )

        result_df = written["domains_with_predictions.xlsx"]
        detail_df = written["domains_prediction_details.xlsx"]
        self.assertEqual(result_df.loc[0, "predicted_class_id"], 0)
        self.assertEqual(result_df.loc[0, "predicted_label"], "Other")
        self.assertEqual(detail_df.loc[0, "predicted_class_id"], 0)
        self.assertEqual(detail_df.loc[0, "predicted_label"], "Other")


if __name__ == "__main__":
    unittest.main()
