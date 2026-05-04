"""UniProt export parsing."""

from __future__ import annotations

import json
from pathlib import Path


def parse_uniprot(unipro_file_path):
    with open(unipro_file_path, "r", encoding="utf-8") as file:
        uniprot_data = json.loads(file.read())
    parsed = {}
    for uniproinfo in uniprot_data:
        for unidata in uniprot_data[uniproinfo]:
            uniproid = unidata["primaryAccession"]
            pdb_list = []
            af_list = []
            annotation_score = unidata.get("annotationScore")
            goannot = {"C": [], "P": [], "F": []}
            if "uniProtKBCrossReferences" in unidata:
                for databaseinfo in unidata["uniProtKBCrossReferences"]:
                    if databaseinfo["database"] == "PDB":
                        chains_value = next((item["value"] for item in databaseinfo["properties"] if item["key"] == "Chains"), None)
                        chain_id = chains_value.split("=")[0] if chains_value else ""
                        pdb_list.append(databaseinfo["id"] + "_" + chain_id)
                    if databaseinfo["database"] == "AlphaFoldDB":
                        af_list.append(databaseinfo["id"])
                    if databaseinfo["database"] == "GO":
                        for goterm in databaseinfo["properties"]:
                            if goterm["key"] == "GoTerm":
                                aspect, term = goterm["value"].split(":", 1)
                                goannot[aspect].append(term)
            comment_text = []
            diseaserelated = ""
            if "comments" in unidata:
                for comment_info in unidata["comments"]:
                    if "texts" in comment_info:
                        comment_text.append(comment_info["texts"][0]["value"])
                    if "commentType" in comment_info and "DISEASE" in comment_info["commentType"]:
                        diseaserelated = "yes"
            if "genes" in unidata:
                if "geneName" in unidata["genes"][0]:
                    gene = unidata["genes"][0]["geneName"]["value"]
                elif "orfNames" in unidata["genes"][0]:
                    gene = unidata["genes"][0]["orfNames"][0]["value"]
                else:
                    gene = "add_from_uni"
            else:
                gene = ""
            uniprot_domain_resrange = []
            if "features" in unidata:
                for feat in unidata["features"]:
                    if feat.get("type") == "Domain":
                        if feat.get("location", {}).get("start", {}).get("value") and feat.get("location", {}).get("end", {}).get("value"):
                            feature_desc = feat.get("description", "Domain")
                            start = feat["location"]["start"]["value"]
                            end = feat["location"]["end"]["value"]
                            uniprot_domain_resrange.append(f"{feature_desc}:{start}-{end}")
            parsed[uniproid] = {
                "description": unidata.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "comment_text": " | ".join(comment_text),
                "uniprot_gene": gene,
                "annotationScore": annotation_score,
                "disease_related": diseaserelated,
                "go_function": "; ".join(goannot["F"]),
                "go_process": "; ".join(goannot["P"]),
                "go_component": "; ".join(goannot["C"]),
                "pdb_list": "; ".join(pdb_list),
                "alphafold_list": "; ".join(af_list),
                "uniprot_domain_resrange": "; ".join(uniprot_domain_resrange),
            }
    return parsed

