import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your chain architecture sheet
df_arch = pd.read_excel("chain_architecture_all_unfiltered.xlsx")

# Filter where num_tm_ig_domains is 1
tm_one = df_arch [df_arch['num_tm_ig_domains'] == 1]

# Get value counts of tm_igdomain_arch
tm_distribution = tm_one['tm_igtype_arch'].value_counts()

print("\nDistribution for tm_igdomain_arch when num_tm_ig_domains == 1")
print(tm_distribution)

# summary for predicted and Igstrand
#sns.set(style="whitegrid")

# Calculate unique domain counts for x-ticks
predicted_ticks = sorted(df_arch["num_predicted_ig_domains"].dropna().unique())
tm_ticks = sorted(df_arch["num_tm_ig_domains"].dropna().unique())

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Plot 1: Predicted
pred_plot = sns.histplot(
    df_arch["num_predicted_ig_domains"],
    bins=[x - 0.5 for x in range(min(predicted_ticks), max(predicted_ticks) + 2)],
    edgecolor="black",
    color="red",
    ax=axes[0]
)
axes[0].set_title("Predicted Ig Domains")
axes[0].set_xlabel("Number of Predicted Ig Domains")
axes[0].set_ylabel("Count")
axes[0].set_xticks(predicted_ticks)

for patch in pred_plot.patches:
    height = patch.get_height()
    if height > 0:
        axes[0].text(patch.get_x() + patch.get_width() / 2, height + 1, f"{int(height)}",
                     ha='center', va='bottom', fontsize=8)

# Plot 2: TM
tm_plot = sns.histplot(
    df_arch["num_tm_ig_domains"],
    bins=[x - 0.5 for x in range(min(tm_ticks), max(tm_ticks) + 2)],
    edgecolor="black",
    color="blue",
    ax=axes[1]
)
axes[1].set_title("TM Ig Domains")
axes[1].set_xlabel("Number of TM Ig Domains")
axes[1].set_xticks(tm_ticks)

for patch in tm_plot.patches:
    height = patch.get_height()
    if height > 0:
        axes[1].text(patch.get_x() + patch.get_width() / 2, height + 1, f"{int(height)}",
                     ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("side_by_side_ig_domain_counts_centered_ticks.png", dpi=300)
plt.show()





# Count of predicted Ig domains
predicted_counts = df_arch["num_predicted_ig_domains"].value_counts().sort_index()
print("\nPredicted Ig Domain Counts:")
print(predicted_counts)
print(f"Total unique chains with predicted domains: {df_arch['num_predicted_ig_domains'].notna().sum()}")

# Count of TM Ig domains
tm_counts = df_arch["num_tm_ig_domains"].value_counts().sort_index()
print("\nTM Ig Domain Counts:")
print(tm_counts)
print(f"Total unique chains with TM domains: {df_arch['num_tm_ig_domains'].notna().sum()}")


# === Helper function ===
def starts_with_IgV_IgC1(arch_string):
    parts = arch_string.split("*")
    return len(parts) >= 3 and parts[0] == "IgV" and parts[2] == "IgC1"

# === Filter based on predicted and TM ===
df_arch["start_IgV_IgC1_pred"] = df_arch["predicted_igtype_arch"].apply(starts_with_IgV_IgC1)
df_arch["start_IgV_IgC1_tm"] = df_arch["tm_igtype_arch"].apply(starts_with_IgV_IgC1)


# Count how many match
count_match_pred = df_arch["start_IgV_IgC1_pred"].sum()
count_match_igstrand= df_arch["start_IgV_IgC1_tm"].sum()

print(f"Chains starting with IgV*linker*IgC1 in predicted: {count_match_pred}")
print(f"Chains starting with IgV*linker*IgC1 in Igstrand: {count_match_igstrand}")

# print(f"Total chains analyzed: {total}")

df_pred = df_arch[df_arch["start_IgV_IgC1_pred"]].copy()
df_tm = df_arch[df_arch["start_IgV_IgC1_tm"]].copy()

# === Overlap analysis ===
pred_ids = set(df_pred["id_chain"])
tm_ids = set(df_tm["id_chain"])

only_pred = pred_ids - tm_ids
only_tm = tm_ids - pred_ids
both_ids = pred_ids & tm_ids

# Print counts
print(f"Only in Predicted: {len(only_pred)}")
print(f"Only in TM: {len(only_tm)}")
print(f"In Both: {len(both_ids)}")

# === Combine domain count data ===
df_domain_counts = pd.concat([
    df_pred[["id_chain", "num_predicted_ig_domains"]].rename(columns={"num_predicted_ig_domains": "count"}).assign(type="Predicted"),
    df_tm[["id_chain", "num_tm_ig_domains"]].rename(columns={"num_tm_ig_domains": "count"}).assign(type="TM")
])


# Prepare data subsets
types = df_domain_counts["type"].unique()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, t in enumerate(types):
    subset = df_domain_counts[df_domain_counts["type"] == t]
    x_ticks = sorted(subset["count"].unique())
    
    ax = axes[i]
    # Use histplot with bins centered on integers
    hist = sns.histplot(
        subset["count"],
        bins=[x - 0.5 for x in range(min(x_ticks), max(x_ticks) + 2)],
        edgecolor="black",
        color="red" if i == 0 else "blue",
        ax=ax
    )
    
    ax.set_title(f"{t} Ig Domains")
    ax.set_xlabel("Number of Ig Domains")
    if i == 0:
        ax.set_ylabel("Count")
    ax.set_xticks(x_ticks)
    
    # Add count labels
    for patch in hist.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

plt.tight_layout()
plt.savefig("separate_ig_domain_igv_IgC1_begin_counts_subplots.png", dpi=300)
plt.show()



# === Combine residues_before_first_domain data ===
df_res_start = pd.concat([
    df_pred[["id_chain", "residues_before_first_domain"]].assign(type="Predicted"),
    df_tm[["id_chain", "residues_before_first_domain"]].assign(type="TM")
])


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

types = df_res_start["type"].unique()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, t in enumerate(types):
    subset = df_res_start[df_res_start["type"] == t]
    data = subset["residues_before_first_domain"]
    
    min_val = int(np.floor(data.min()))
    max_val = int(np.ceil(data.max()))
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)  # bin edges
    
    counts, _ = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Print the bin centers and counts for this type
    print(f"\n{t} - Residues Before First Domain Distribution:")
    for center, count in zip(bin_centers, counts):
        if count > 0:
            print(f"  Residues: {center:.1f}, Count: {count}")
    
    ax = axes[i]
    hist = sns.histplot(
        data,
        bins=bins,
        edgecolor="black",
        color="red" if i == 0 else "blue",
        ax=ax
    )
    
    ax.set_title(f"{t} - Residues Before First Domain")
    ax.set_xlabel("Residues Before First Domain")
    if i == 0:
        ax.set_ylabel("Count")
    
    valid_centers = bin_centers[counts > 0]
    ax.set_xticks(valid_centers)
    
    for patch in hist.patches:
        height = patch.get_height()
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.5,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

plt.tight_layout()
plt.savefig("side_by_side_residues_before_first_centered_bins.png", dpi=300)
plt.show()


# === Plot: Residues Before First Domain ===
plt.figure(figsize=(7, 4))
sns.histplot(data=df_res_start, x="residues_before_first_domain", hue="type", multiple="dodge", bins=30, edgecolor="black")
plt.title("Residues Before First Domain (Starting with IgV→IgC1)")
plt.xlabel("Residues Before First Domain")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("combined_residues_before_first.png", dpi=300)
plt.show()
