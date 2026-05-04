"""Hydrogen bond parsing."""

from __future__ import annotations

import json
from typing import Dict, Set

from Bio.PDB import Polypeptide


def hbond_icn3d_parser(file_path: str) -> Dict[str, Set[str]]:
    hbond_dict: Dict[str, Set[str]] = {}
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    for entry in data["bondCnt"]:
        if entry["cntHbond"] <= 0:
            continue
        res1_parts = entry["res1"].split("_")
        resid1_aa = f"{res1_parts[2]}_{Polypeptide.protein_letters_3to1.get(res1_parts[3], 'X')}"
        hbond_dict.setdefault(resid1_aa, set())
        for interaction in entry["res2"].split():
            if "hbond" not in interaction or "main,main" not in interaction:
                continue
            target_res = interaction.split(":")[0]
            target_parts = target_res.split("_")
            target_resid_aa = f"{target_parts[2]}_{Polypeptide.protein_letters_3to1.get(target_parts[3], 'X')}"
            hbond_dict[resid1_aa].add(target_resid_aa)
            hbond_dict.setdefault(target_resid_aa, set()).add(resid1_aa)
    return hbond_dict

