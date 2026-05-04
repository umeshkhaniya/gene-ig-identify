"""Excel helpers that preserve row order and columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_excel(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)


def write_excel(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path


def prediction_output_path(excel_path: str | Path) -> Path:
    path = Path(excel_path)
    return path.with_name(f"{path.stem}_with_predictions.xlsx")

