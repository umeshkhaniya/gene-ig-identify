"""Prediction workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import DEFAULT_CONFIDENCE_THRESHOLD
from ..io.artifacts import ensure_artifact_dir, load_torch
from ..io.excel import prediction_output_path, write_excel
from ..io.tables import load_table, normalize_domain_table
from ..labels import REVERSE_LABEL_MAPPING
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)


def _validate_model_artifacts(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    required = ["best_graph_model.pth", "best_hyperparameters.json", "model_config.json"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}: {', '.join(missing)}")


def _validate_graph_objects(graphs) -> None:
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("Expected a non-empty list of graph objects.")
    required_attrs = ("unique_name_file", "x", "edge_index", "edge_attr")
    first_graph = graphs[0]
    missing = [attr for attr in required_attrs if not hasattr(first_graph, attr)]
    if missing:
        raise ValueError(f"Graphs file is missing required graph attributes: {', '.join(missing)}")


def _optional_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _expected_graph_name(row: pd.Series) -> str | None:
    if pd.isna(row.get("igdomain_res_range")):
        return None
    begin_res, end_res = str(row["igdomain_res_range"]).split("_")
    template_name = _optional_string(row.get("refpdbname"))
    ig_type = _optional_string(row.get("ig_type"))
    parts = [row["pdb"], row["chainid"], begin_res, end_res]
    if template_name:
        parts.append(template_name)
    if ig_type:
        parts.append(ig_type)
    return "_".join(parts)


def run_excel_predictions(config, graphs_file: Path, excel_file: Path, model_dir: Path, output_dir: Path) -> Path:
    from ..models.inference import load_model, predict_graphs, resolve_device

    output_dir = ensure_artifact_dir(output_dir)
    _validate_model_artifacts(model_dir)
    if not graphs_file.exists():
        raise FileNotFoundError(f"Graphs file not found: {graphs_file}")
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")
    graphs = load_torch(graphs_file)
    _validate_graph_objects(graphs)
    excel_df = load_table(excel_file)
    normalized_df = normalize_domain_table(excel_df)
    if graphs and hasattr(graphs[0], "source_row_index"):
        graphs = sorted(graphs, key=lambda graph: int(graph.source_row_index))
    if len(graphs) != len(normalized_df):
        raise ValueError(f"Row alignment issue: {len(normalized_df)} Excel rows but {len(graphs)} graph entries.")
    device = resolve_device(str(config.runtime.get("device", "auto")))
    model, best_params = load_model(model_dir, device)
    all_preds, all_probs, _, graph_names = predict_graphs(graphs, model, best_params["batch_size"], device)
    result_df = excel_df.copy()
    predicted_labels = []
    predicted_ids = []
    confidences = []
    prediction_rows = []
    for idx, graph_name in enumerate(graph_names):
        if hasattr(graphs[idx], "source_row_index") and int(graphs[idx].source_row_index) != idx:
            raise ValueError("Row alignment issue: graph source_row_index does not match Excel row order.")
        expected_graph_name = _expected_graph_name(normalized_df.iloc[idx])
        if expected_graph_name and graph_name != expected_graph_name:
            raise ValueError(
                "Row alignment issue: graph order does not match Excel row order at "
                f"row {idx}. Expected {expected_graph_name}, got {graph_name}."
            )
        top_idx = int(all_preds[idx])
        confidence = float(all_probs[idx][top_idx])
        predicted_label = REVERSE_LABEL_MAPPING[top_idx]
        if confidence < float(config.model.get("default_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)):
            predicted_label = "Other"
        predicted_labels.append(predicted_label)
        predicted_ids.append(top_idx)
        confidences.append(confidence)
        prediction_rows.append({
            "row_index": idx,
            "graph_name": graph_name,
            "predicted_label": predicted_label,
            "predicted_class_id": top_idx,
            "prediction_confidence": confidence,
        })
    result_df["predicted_label"] = predicted_labels
    result_df["predicted_class_id"] = predicted_ids
    result_df["prediction_confidence"] = confidences
    output_path = output_dir / prediction_output_path(excel_file).name
    write_excel(result_df, output_path)
    write_excel(pd.DataFrame(prediction_rows), output_dir / f"{excel_file.stem}_prediction_details.xlsx")
    LOGGER.info("Saved row-preserving Excel predictions to %s", output_path)
    return output_path
