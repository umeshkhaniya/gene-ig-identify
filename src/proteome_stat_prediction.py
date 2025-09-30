import pandas as pd




# Function to process PDB dataset
def ml_prediction_filter(file_path, prob_cutoff):
    """Reads PDB data, filters by TM score, and extracts PDB and chain IDs."""
    df = pd.read_excel(file_path)

    df['Probability'] = df['Probability'].astype(float).round(2)

    df = df[df['Probability'] >= prob_cutoff]
    
    # Extract PDB ID and chain ID
    df[["pdb", "chainid"]] = df["pdbid_chain"].str.split("_", expand=True)
    
    # Keep only rows where 'chainid' is a single character
    df_filtered = df[df["chainid"].str.len() == 1]

    print(f"ML prediction {file_path}: {df.shape} → Filtered: {df_filtered.shape}, Unique PDBs: {df_filtered['pdb'].nunique()}")
    return df_filtered


ml_prediction_filter('merged_result1.xlsx', prob_cutoff = 0.90)

# how many predictions matches based on TM score filter?


# Function to process PDB dataset
def filter_tm_score(file_path, tm_threshold):
    """Reads PDB data, filters by TM score, and extracts PDB and chain IDs."""
    df = pd.read_excel(file_path)

    df['score'] = df['score'].astype(float).round(2)

    df = df[df['score'] >= tm_threshold]
    
    # Extract PDB ID and chain ID
    df[["pdb", "chainid"]] = df["pdbid_chain"].str.split("_", expand=True)
    
    # Keep only rows where 'chainid' is a single character
    df_filtered = df[df["chainid"].str.len() == 1]

    print(f"TM filter:{file_path}: {df.shape} → Filtered: {df_filtered.shape}, Unique PDBs: {df_filtered['pdb'].nunique()}")
    return df_filtered

df_tm_filter = filter_tm_score('merged_result1.xlsx', tm_threshold = 0.8)

# df_tm_filter.to_excel("tm_filter_0.9.xlsx")

#To compare the distribution of top_predicted_ml for each ig_type (e.g., how many times top_predicted_ml equals something when ig_type is "IgI"),
# Group by ig_type and count occurrences of top_predicted_ml
count_df = df_tm_filter.groupby('ig_type')['Top_Predicted_Label'].value_counts().unstack(fill_value=0)

# Add total count of each ig_type
count_df['Total'] = count_df.sum(axis=1)

# Calculate percentage distribution
percentage_df = count_df.div(count_df['Total'], axis=0) * 100

# Optional: round percentages for clarity
percentage_df = percentage_df.round(2)


# Combine count and percentage, but show blank if count is zero
combined_df = count_df.copy()

for col in count_df.columns:
    if col == 'Total':
        # Show only total count, no percentage
        combined_df[col] = count_df[col].astype(str)
    else:
        # Show count and percentage, or blank if count is 0
        combined_df[col] = [
            f"{count} ({percentage_df.loc[idx, col]:.1f}%)" if count > 0 else ""
            for idx, count in count_df[col].items()
        ]




output_file = "ig_type_vs_top_predicted_ml2.xlsx"
combined_df.to_excel(output_file)

print(combined_df)



