"""Evaluate row-preserving prediction Excel output against an ig_type column."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_prediction_excel.py",
        description="Calculate label agreement for an Excel prediction file with ig_type and predicted_label columns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-file",
        default="output/input_data_with_predictions.xlsx",
        help="Prediction Excel file created by `gene-ig-identify predict dataset`.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictions_file = Path(args.predictions_file)
    df = pd.read_excel(predictions_file)

    required_columns = {"ig_type", "predicted_label"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{predictions_file} is missing required columns: {', '.join(missing)}")

    df = df[df["ig_type"].notna()].copy()
    accuracy = (df["ig_type"].astype(str) == df["predicted_label"].astype(str)).mean()

    print(f"Rows evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(pd.crosstab(df["ig_type"], df["predicted_label"], rownames=["true"], colnames=["predicted"]))


if __name__ == "__main__":
    main()
