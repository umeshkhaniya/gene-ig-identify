"""Evaluate saved held-out test graphs from a trained model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_test_graphs.py",
        description="Create a readable held-out test report from saved test_graphs.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="results/models",
        help="Directory containing the trained model artifacts and saved test split.",
    )
    parser.add_argument(
        "--test-graphs",
        help="Path to test_graphs.pt. Defaults to <model-dir>/test_graphs.pt.",
    )
    parser.add_argument(
        "--test-labels",
        help="Path to test_labels.pt. Defaults to <model-dir>/test_labels.pt.",
    )
    parser.add_argument(
        "--output-csv",
        help="CSV output path. Defaults to <model-dir>/heldout_test_predictions.csv.",
    )
    parser.add_argument(
        "--output-excel",
        help="Excel output path. Defaults to <model-dir>/heldout_test_predictions.xlsx.",
    )
    parser.add_argument(
        "--classification-report-csv",
        help="Classification report CSV path. Defaults to <model-dir>/heldout_classification_report.csv.",
    )
    parser.add_argument(
        "--classification-report-excel",
        help="Classification report Excel path. Defaults to <model-dir>/heldout_classification_report.xlsx.",
    )
    parser.add_argument(
        "--confusion-matrix-csv",
        help="Confusion matrix CSV path. Defaults to <model-dir>/heldout_confusion_matrix.csv.",
    )
    parser.add_argument(
        "--confusion-matrix-excel",
        help="Confusion matrix Excel path. Defaults to <model-dir>/heldout_confusion_matrix.xlsx.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser


def _label_ids_from_graphs_or_file(graph_labels, test_labels_file: Path) -> list[int]:
    from gene_ig_identify.io.artifacts import load_torch

    if len(graph_labels):
        return [int(label) for label in graph_labels]
    test_labels = load_torch(test_labels_file)
    return [int(label) for label in test_labels]


def _format_score(value) -> float | str:
    if pd.isna(value):
        return ""
    return round(float(value), 2)


def _format_support(value, include_support: bool) -> int | str:
    if not include_support or pd.isna(value):
        return ""
    return int(value)


def _classification_report_table(report: dict) -> pd.DataFrame:
    rows = []
    for row_name, values in report.items():
        if row_name == "accuracy":
            rows.append(
                {
                    "label": row_name,
                    "precision": "",
                    "recall": "",
                    "f1-score": _format_score(values),
                    "support": "",
                }
            )
            continue

        include_support = row_name not in {"macro avg", "weighted avg"}
        rows.append(
            {
                "label": row_name,
                "precision": _format_score(values.get("precision")),
                "recall": _format_score(values.get("recall")),
                "f1-score": _format_score(values.get("f1-score")),
                "support": _format_support(values.get("support"), include_support),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def _experiment_root_from_model_dir(model_dir: Path) -> Path | None:
    if model_dir.name == "models" and model_dir.parent.parent.name == "experiments":
        return model_dir.parent
    return None


def _default_predictions_dir(model_dir: Path) -> Path:
    experiment_root = _experiment_root_from_model_dir(model_dir)
    return experiment_root / "predictions" if experiment_root else model_dir


def _default_metrics_dir(model_dir: Path) -> Path:
    experiment_root = _experiment_root_from_model_dir(model_dir)
    return experiment_root / "metrics" if experiment_root else model_dir


def main(argv: list[str] | None = None) -> None:
    from gene_ig_identify.io.artifacts import load_torch
    from gene_ig_identify.models.inference import load_model_artifacts, predict_graphs, resolve_device

    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    test_graphs_file = Path(args.test_graphs) if args.test_graphs else model_dir / "test_graphs.pt"
    test_labels_file = Path(args.test_labels) if args.test_labels else model_dir / "test_labels.pt"
    predictions_dir = _default_predictions_dir(model_dir)
    metrics_dir = _default_metrics_dir(model_dir)
    output_csv = Path(args.output_csv) if args.output_csv else predictions_dir / "heldout_test_predictions.csv"
    output_excel = Path(args.output_excel) if args.output_excel else predictions_dir / "heldout_test_predictions.xlsx"
    report_csv = (
        Path(args.classification_report_csv)
        if args.classification_report_csv
        else metrics_dir / "heldout_classification_report.csv"
    )
    report_excel = (
        Path(args.classification_report_excel)
        if args.classification_report_excel
        else metrics_dir / "heldout_classification_report.xlsx"
    )
    matrix_csv = (
        Path(args.confusion_matrix_csv)
        if args.confusion_matrix_csv
        else metrics_dir / "heldout_confusion_matrix.csv"
    )
    matrix_excel = (
        Path(args.confusion_matrix_excel)
        if args.confusion_matrix_excel
        else metrics_dir / "heldout_confusion_matrix.xlsx"
    )

    graphs = load_torch(test_graphs_file)
    device = resolve_device(args.device)
    model, best_params, label_space = load_model_artifacts(model_dir, device)
    reverse_label_mapping = label_space.reverse_label_mapping
    preds, probs, graph_labels, graph_names = predict_graphs(
        graphs,
        model,
        best_params["batch_size"],
        device,
    )

    true_ids = _label_ids_from_graphs_or_file(graph_labels, test_labels_file)
    if len(true_ids) != len(preds):
        raise ValueError(f"Expected {len(preds)} labels, found {len(true_ids)} in the held-out test data.")

    rows = []
    for idx, pred_id in enumerate(preds):
        pred_id = int(pred_id)
        true_id = int(true_ids[idx])
        rows.append(
            {
                "graph_name": graph_names[idx],
                "true_class_id": true_id,
                "true_label": reverse_label_mapping[true_id],
                "predicted_class_id": pred_id,
                "predicted_label": reverse_label_mapping[pred_id],
                "prediction_confidence": float(probs[idx][pred_id]),
                "correct": pred_id == true_id,
            }
        )

    df = pd.DataFrame(rows)
    accuracy = df["correct"].mean()
    labels = sorted(set(df["true_class_id"]) | set(df["predicted_class_id"]))
    label_names = [reverse_label_mapping[label] for label in labels]
    report = classification_report(
        df["true_class_id"],
        df["predicted_class_id"],
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = _classification_report_table(report)
    confusion_df = pd.DataFrame(
        confusion_matrix(df["true_class_id"], df["predicted_class_id"], labels=labels),
        index=label_names,
        columns=label_names,
    )
    confusion_df.index.name = "true"
    confusion_df.columns.name = "predicted"

    for output_path in (output_csv, output_excel, report_csv, report_excel, matrix_csv, matrix_excel):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)
    report_df.to_csv(report_csv)
    report_df.to_excel(report_excel)
    confusion_df.to_csv(matrix_csv)
    confusion_df.to_excel(matrix_excel)

    print(f"Held-out test rows: {len(df)}")
    print(f"Held-out test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report_df)
    print("Confusion matrix:")
    print(confusion_df)
    print(f"Saved predictions CSV: {output_csv}")
    print(f"Saved predictions Excel: {output_excel}")
    print(f"Saved classification report CSV: {report_csv}")
    print(f"Saved classification report Excel: {report_excel}")
    print(f"Saved confusion matrix CSV: {matrix_csv}")
    print(f"Saved confusion matrix Excel: {matrix_excel}")


if __name__ == "__main__":
    main()
