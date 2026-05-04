"""Postprocessing workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..integrations.uniprot import parse_uniprot
from ..io.excel import read_excel, write_excel


def merge_predictions_with_annotations(config, excel_file: Path, predictions_file: Path, uniprot_json: Path, output_file: Path | None = None) -> Path:
    df1 = read_excel(excel_file)
    df2 = read_excel(predictions_file)
    uniprot_annotation = parse_uniprot(uniprot_json)
    if "pdbid_chain" in df1.columns:
        id_column = "pdbid_chain"
    elif "id_chain" in df1.columns:
        id_column = "id_chain"
    else:
        raise ValueError("Expected pdbid_chain or id_chain in Excel file.")
    df1["uniprotid"] = df1[id_column].astype(str).str.split("_").str[0]
    dict_uniprot_annotation = pd.DataFrame.from_dict(uniprot_annotation, orient="index").reset_index().rename(columns={"index": "uniprotid"})
    merged_base = df1.merge(dict_uniprot_annotation, on="uniprotid", how="left")
    merged = pd.concat([merged_base.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
    target = output_file or excel_file.with_name(f"{excel_file.stem}_merged_predictions.xlsx")
    return write_excel(merged, target)


def create_arch_string_with_linker(values, linkers):
    parts = []
    for i in range(len(values) - 1):
        parts.append(f"{values[i]}*{linkers[i]}")
    parts.append(values[-1])
    return "*".join(parts)


def create_chain_architecture_summary(input_file: Path, output_file: Path, skip_single_domain: bool = False) -> Path:
    df = pd.read_excel(input_file)
    if "start" not in df.columns or "end" not in df.columns:
        df[["start", "end"]] = df["igdomain_res_range"].str.split("_", expand=True).astype(int)
    df = df.sort_values(["id_chain", "start", "end"]).copy()
    df["start2"] = df.groupby("id_chain")["start"].shift(-1)
    df["Linker length"] = df["start2"] - df["end"] - 1
    df["Linker length"] = df["Linker length"].fillna(0).astype(int)
    if skip_single_domain:
        df = df[df.groupby("id_chain")["id_chain"].transform("count") > 1]

    def summarize_chain(group):
        id_chain = group.name
        uniprot_id = id_chain.split("_")[0]
        predicted_clean = group[~group["predicted_label"].isin(["Other", "JellyRoll"])] if "predicted_label" in group.columns else group
        return pd.Series({
            "gene_name": group["gene_name"].iloc[0] if "gene_name" in group.columns else "",
            "icn3d_link": f'=HYPERLINK("https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={uniprot_id}&command=ig+refnum+on", "view_structure")',
            "num_predicted_ig_domains": len(predicted_clean),
            "num_tm_ig_domains": len(group),
            "residues_before_first_domain": group["start"].iloc[0] - 1,
            "predicted_igtype_arch": create_arch_string_with_linker(group["predicted_label"].tolist() if "predicted_label" in group.columns else group["predicted_igtype"].tolist(), group["Linker length"].tolist()),
            "tm_igtype_arch": create_arch_string_with_linker(group["tm_igtype"].tolist() if "tm_igtype" in group.columns else group["ig_type"].tolist(), group["Linker length"].tolist()),
            "igdomain_res_range_arch": "*".join(str(x) for x in group["igdomain_res_range"].tolist()),
            "refpdbname_arch": "*".join(group["refpdbname"].astype(str).tolist()) if "refpdbname" in group.columns else "",
            "pdb": group["pdb"].iloc[0] if "pdb" in group.columns else id_chain.split("_")[0],
        })

    arch_df = df.groupby("id_chain", group_keys=False, observed=True).apply(summarize_chain, include_groups=False).reset_index()
    return write_excel(arch_df, output_file)

