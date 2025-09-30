# Here I didn't remove the negative linker and so on


import pandas as pd

def create_arch_string_with_linker(values, linkers):
    """Concatenate domain names and linker lengths (including negatives)."""
    parts = []
    for i in range(len(values) - 1):
        parts.append(f"{values[i]}*{linkers[i]}")
    parts.append(values[-1])
    return "*".join(parts)

def create_chain_architecture_summary(input_file, output_file, skip_single_domain=False):
    # Load the input Excel file
    df = pd.read_excel(input_file)

    # Extract start and end if not already present
    if "start" not in df.columns or "end" not in df.columns:
        df[["start", "end"]] = df["igdomain_res_range"].str.split("_", expand=True).astype(int)

    # Sort by id_chain, start, and end
    df = df.sort_values(["id_chain", "start", "end"]).copy()

    # Compute linker length: start of next domain - end of current domain - 1
    df["start2"] = df.groupby("id_chain")["start"].shift(-1)
    df["Linker length"] = df["start2"] - df["end"] - 1
    df["Linker length"] = df["Linker length"].fillna(0).astype(int)


    #Print rows with negative linker length
    negative_linkers = df[df["Linker length"] < 0]
    if not negative_linkers.empty:
        print("⚠️ Negative linker details:")
        print(negative_linkers[["id_chain", "Linker length"]].to_string(index=False))


    # Optional: Skip chains with only one domain
    if skip_single_domain:
        df = df[df.groupby("id_chain")["id_chain"].transform("count") > 1]

    def summarize_chain(group):
        id_chain = group.name
        uniprot_id = id_chain.split("_")[0]
        predicted_clean = group[~group["predicted_igtype"].isin(["Other", "JellyRoll"])]

        return pd.Series({
            "gene_name": group["gene_name"].iloc[0],
            "icn3d_link": f'=HYPERLINK("https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={uniprot_id}&command=ig+refnum+on", "view_structure")',
            "uniprot_link": f'=HYPERLINK("https://www.uniprot.org/uniprotkb/{uniprot_id}/entry", "view_uniprot")',
            "num_predicted_ig_domains": len(predicted_clean),
            "num_tm_ig_domains": len(group),
            "residues_before_first_domain": group["start"].iloc[0] - 1,
            "predicted_igtype_arch": create_arch_string_with_linker(group["predicted_igtype"].tolist(), group["Linker length"].tolist()),
            "tm_igtype_arch": create_arch_string_with_linker(group["tm_igtype"].tolist(), group["Linker length"].tolist()),
            "cdd_annotation_arch": "*".join(str(x) for x in group["cdd_annotation"].tolist()),
            "igdomain_res_range_arch": "*".join(str(x) for x in group["igdomain_res_range"].tolist()),
            "tm_score_arch": (
                    float(f"{group['tm_score'].iloc[0]:.1f}") if len(group) == 1 
                    else "*".join(f"{x:.1f}" for x in group["tm_score"])
                ),

            "refpdbname_arch": "*".join(group["refpdbname"].tolist()),
            "pdb": group["pdb"].iloc[0],
        })


    # # Group by id_chain and summarize
    # def summarize_chain(group):
    #     id_chain = group.name
    #     uniprot_id = id_chain.split("_")[0]
    #     predicted_clean = group[~group["predicted_igtype"].isin(["Other", "JellyRoll"])]
    #     uniprot_id = group["id_chain"].iloc[0].split("_")[0]
    #     return pd.Series({
    #         "uniprot_id": uniprot_id,
    #         "gene_name": group["gene_name"].iloc[0],
    #         "icn3d_link": f'=HYPERLINK("https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={uniprot_id}&command=ig+refnum+on", "view_structure")',
    #         "uniprot_link": group["uniprot_link"].iloc[0],
    #         "num_predicted_ig_domains":  len(predicted_clean),
    #         "num_tm_ig_domains": len(group),
    #         "residues_before_first_idomain": group["start"].iloc[0] - 1,
    #         "predicted_igtype_arch": create_arch_string_with_linker(group["predicted_igtype"].tolist(), group["Linker length"].tolist()),
    #         "tm_igtype_arch": create_arch_string_with_linker(group["tm_igtype"].tolist(), group["Linker length"].tolist()),
    #         "cdd_annotation_arch": ":".join(str(x) for x in group["cdd_annotation"].tolist()),
    #         "igdomain_res_range_arch": ":".join(str(x) for x in group["igdomain_res_range"].tolist()),
    #         "tm_score_arch": "_".join(f"{x:.1f}" for x in group["tm_score"]),
    #         "refpdbname_arch": "_".join(group["refpdbname"].tolist()),
    #         "pdb": group["pdb"].iloc[0],
    #     })


    arch_df = (
    df.groupby("id_chain", group_keys=False, observed=True)
    .apply(summarize_chain, include_groups=False)
    .reset_index()
    )


    #arch_df = df.groupby("id_chain", group_keys=False).apply(summarize_chain).reset_index()

    # Save to Excelç
    #arch_df.to_excel(output_file, index=False)
    print(f"✅ Saved chain architecture summary to: {output_file}")

    #print(arch_df["predicted_igtype_arch"])

if __name__ == "__main__":
    input_file = "merged1_result_uniquegenes_human.xlsx"
    output_file = "chain_architecture_all_unfiltered.xlsx"
    create_chain_architecture_summary(input_file, output_file, skip_single_domain=False)


