# This will find the the chain architecture based on consecutive domains.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your Excel or CSV file
df_all = pd.read_excel("merged1_result_uniquegenes_human.xlsx")  # or pd.read_csv("your_file.csv")

# === STATS BEFORE FILTERING ===
total_ids = df_all["id_chain"].nunique()
total_domains = len(df_all)

# Rows to be removed (Other or JellyRoll)
excluded = df_all[df_all["predicted_igtype"].isin(["Other", "JellyRoll"])]
excluded_ids = excluded["id_chain"].nunique()
excluded_domains = len(excluded)

print("=== Domain Statistics ===")
print(f"Total unique id_chain : {total_ids}")
print(f"Total domains before filtering: {total_domains}")
print(f"Domains removed (Other/JellyRoll): {excluded_domains}")
print(f"Unique id_chain affected by removal: {excluded_ids}")


# --- Filter out rows where predicted_igtype is Other or JellyRoll ---
df = df_all[~df_all["predicted_igtype"].isin(["Other", "JellyRoll"])].copy()


# Plot both distributions
plt.figure(figsize=(8, 5))

# sns.histplot(all_scores, bins=30, color="blue", label="All domains", kde=True, stat="count", alpha=0.5)
# sns.histplot(filtered_scores, bins=30, color="red", label="Filtered (no Other/JellyRoll)", kde=True, stat="count", alpha=0.5)

sns.histplot(df_all['tm_score'], bins=30, color="blue", label="IgStrand", kde=True, stat="count", alpha=0.6, edgecolor="blue")
sns.histplot(df['tm_score'], bins=30, color="red", label="Predicted", kde=True, stat="count", alpha=0.6, edgecolor="red")
plt.legend(facecolor='white', framealpha=1, edgecolor='black')

plt.title("TM-score Distribution IgStrand Vs ML")
plt.xlabel("TM-score")
plt.ylabel("Count")
plt.xlim(0.4, 1)

plt.tight_layout()
plt.savefig("tm_score_comparison.png", dpi=300)
plt.show()




# Stats after filtering
remaining_ids = df["id_chain"].nunique()
remaining_domains = len(df)

print(f"Remaining unique id_chain after filter: {remaining_ids}")
print(f"Remaining domains after filter: {remaining_domains}")



# === Unique Chains with Only One Domain After Filtering ===

# Count how many domains per id_chain after filtering
domain_counts = df["id_chain"].value_counts()

# Keep only those with exactly one domain
single_ig_chains = domain_counts[domain_counts == 1].index

# Filter the original df for these chains
single_ig_df = df[df["id_chain"].isin(single_ig_chains)].copy()

# === Distribution by predicted Ig type ===
predicted_dist = single_ig_df["predicted_igtype"].value_counts()
print("\n=== Distribution of Single Ig Domains (Predicted) ===")
print(predicted_dist)

domain_counts = df["id_chain"].value_counts()
single_igs = (domain_counts == 1).sum()

print(f"\nNumber of unique id_chain with exactly one Ig domain after filtering: {single_igs}")



# Extract start and end from igdomain_res_range
df[["start", "end"]] = df["igdomain_res_range"].str.split("_", expand=True).astype(int)

# Sort by id_chain, start and end
df = df.sort_values(["id_chain", "start", "end"]).reset_index(drop=True)

# Create icn3d link
df["uniprot_id"] = df["id_chain"].apply(lambda x: x.split("_")[0])


df["icn3d_link"] = df["uniprot_id"].apply(
    lambda x: f'=HYPERLINK("https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={x}&command=ig+refnum+on","view_structure")'
)



df['uniprot_link'] = df['uniprot_id'].apply(
    lambda uniprotid: f'=HYPERLINK("https://www.uniprot.org/uniprotkb/{uniprotid}/entry", "view_uniprot")'
)


# Shift to get domain2 and paired values
df["domain2"] = df.groupby("id_chain")["igdomain_res_range"].shift(-1)
df["predicted_igtype2"] = df.groupby("id_chain")["predicted_igtype"].shift(-1)
df["tm_igtype2"] = df.groupby("id_chain")["tm_igtype"].shift(-1)
df["cdd_annot2"] = df.groupby("id_chain")["cdd_annotation"].shift(-1)
df["tm_score2"] = df.groupby("id_chain")["tm_score"].shift(-1)
df["refpdbname2"] = df.groupby("id_chain")["refpdbname"].shift(-1)
df["start2"] = df.groupby("id_chain")["start"].shift(-1)
df["end1"] = df["end"]

# Rename current row columns
df = df.rename(columns={
    "igdomain_res_range": "domain1",
    "predicted_igtype": "predicted_igtype1",
    "tm_igtype": "tm_igtype1",
    "tm_score": "tm_score1",
    "refpdbname": "refpdbname1",
    "cdd_annotation": "cdd_annot1"
})

# Drop rows where domain2 is missing (last domain in each chain)
df = df.dropna(subset=["domain2"])

# Calculate linker length
df["Linker length"] = df["start2"] - df["end1"] - 1 # not including both

# Identify negative linker pairs
negative_linkers = df[df["Linker length"] < 0]

# Count total negative domain pairs
neg_pair_count = len(negative_linkers)

# Count how many unique id_chains are affected
neg_id_count = negative_linkers["id_chain"].nunique()

print(f"\n=== Negative Linker Tandem Domain Pairs ===")
print(f"Total negative linker domain pairs: {neg_pair_count}")




# Remove rows where linker is negative. Here I removed negative linker.

print("The tandem igs whose length negative  is removed")

df = df[df["Linker length"] >= 0].copy()

# Add domain pair number
df["tandem_igpair"] = df.groupby("id_chain").cumcount() + 1

# Calculate average linker length per chain, rounded to 2 decimals
df["Avg linker"] = df.groupby("id_chain")["Linker length"].transform("mean").round(2)

# add genes name.
# Map genes_name from original dataframe to df_out using id_chain
id_to_gene = df_all[["id_chain", "gene_name"]].drop_duplicates().set_index("id_chain")["gene_name"]
df["gene_name"] = df["id_chain"].map(id_to_gene)

print(list(df.columns))

# Final output columns
df_out = df[[
    "id_chain", "tandem_igpair", "icn3d_link", "uniprot_link", "gene_name",
    "domain1", "domain2",
    "predicted_igtype1", "predicted_igtype2",
    "tm_igtype1", "tm_igtype2",
    "cdd_annot1", "cdd_annot2",
    "tm_score1", "tm_score2",
    "refpdbname1", "refpdbname2",
    "Linker length", "Avg linker"   
]]



# === Domain Pairing Summary ===
paired_ids = df_out["id_chain"].nunique()
total_pairs = len(df_out)

print("\n=== Domain Pairing Summary  after removing negative linkers===")
print(f"Number of unique id_chain with at least one tandem Igs: {paired_ids}")
print(f"Total number of tandem Igs: {total_pairs}")


# Save output file
df_out.to_excel("final_paired_igdomains_try.xlsx", index=False)

# === STATISTICS ===

df_stat = df_out.copy()
df_stat["pred_pair"] = df_stat["predicted_igtype1"] + "-" + df_stat["predicted_igtype2"]
pair_counts = df_stat["pred_pair"].value_counts()

# Group by predicted pair and calculate linker statistics
linker_stats = df_stat.groupby("pred_pair")["Linker length"].agg(
    avg_linker="mean",
    max_linker="max",
    min_linker="min",
    count="count"
).round(2).reset_index()

# Display the table
print("\n=== Linker Statistics by Predicted Ig Domain Pair ===")
print(linker_stats)

# Save to Excel
linker_stats.to_excel("linker_stats_by_predicted_pair.xlsx", index=False)



# Save pair counts to Excel
pair_counts.to_excel("domain_pair_counts.xlsx", header=["count"])
print("\n=== Domain Pair Type Counts (Predicted Types) ===\n")
print(pair_counts)

# Plot top 10 pairs only
top10 = pair_counts.head(10)



plt.figure(figsize=(10, 5))
top10.plot(kind="bar", color="steelblue")
plt.title("Top 10 Predicted  Tandem Ig ")
plt.ylabel("Count")
plt.xlabel("Domain Pair")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("top10_tandem_domain_pair_histogram.png", dpi=300)
plt.show()
