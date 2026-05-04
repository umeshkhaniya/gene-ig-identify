"""Table schema normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class NormalizedRow:
    pdb: str
    chain: str
    ig_type: str | None
    igdomain_res_range: str
    refpdbname: str | None
    original_index: int


def load_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(input_path, sep="\t")
    raise ValueError(f"Unsupported table format: {input_path}. Use .xlsx, .xls, .csv, .tsv, or .txt.")


def _first_present(row: pd.Series, keys: list[str], default: str | None = None) -> str | None:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return default


def normalize_domain_table(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "pdbid_chain" in normalized.columns:
        split = normalized["pdbid_chain"].astype(str).str.split("_", expand=True)
        normalized["pdb"] = split[0].str.upper()
        normalized["chainid"] = split[1]
    elif {"pdb", "chainid"}.issubset(normalized.columns):
        normalized["pdb"] = normalized["pdb"].astype(str).str.upper()
        normalized["chainid"] = normalized["chainid"].astype(str)
        normalized["pdbid_chain"] = normalized["pdb"] + "_" + normalized["chainid"]
    elif {"id_chain"}.issubset(normalized.columns):
        split = normalized["id_chain"].astype(str).str.split("_", expand=True)
        normalized["pdb"] = split[0].str.upper()
        normalized["chainid"] = split[1]
        normalized["pdbid_chain"] = normalized["id_chain"]
    else:
        raise ValueError("Input table must contain either pdbid_chain, id_chain, or pdb/chainid columns.")

    if "ig_type" not in normalized.columns and "igtype" in normalized.columns:
        normalized["ig_type"] = normalized["igtype"]
    if "ig_type" not in normalized.columns:
        normalized["ig_type"] = None
    if "refpdbname" not in normalized.columns:
        normalized["refpdbname"] = None
    if "igdomain_res_range" not in normalized.columns:
        raise ValueError("Input table must contain igdomain_res_range.")
    normalized["_row_order"] = range(len(normalized))
    return normalized


def preserve_row_order(df: pd.DataFrame) -> pd.DataFrame:
    if "_row_order" in df.columns:
        return df.sort_values("_row_order", kind="stable").drop(columns="_row_order")
    return df
