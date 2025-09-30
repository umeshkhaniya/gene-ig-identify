# This will find the how many genes any based on probabilities?
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

    print(f"ML prediction with probability:{prob_cutoff} {file_path}: {df.shape} → Filtered: {df_filtered.shape}, Unique PDBs: {df_filtered['pdb'].nunique()}")
    return df_filtered


ml_res = ml_prediction_filter('merged_result_uniquegenes.xlsx', prob_cutoff = 0.90)



# find IgC1 with atleast one Ig and probability cutoff 0.5

# Step 1: Filter for rows with 'IgC1'
igc1_df = ml_res[ml_res['Top_Predicted_Label'] == 'IgC1'].copy()

# Step 2: Count how many values have 'IgC1'
count_igc1 = len(igc1_df)


igc1_df['pdbid'] = igc1_df['pdbid_chain'].str.split('_').str[0]
unique_pdbids = igc1_df['pdbid'].nunique()


print("Number of rows with IgC1:", count_igc1)
print("Number of unique pdbids with IgC1:", unique_pdbids)


# Step 2: Find all pdbids that have at least one IgC1 domain

ml_res_pdb = ml_res.copy()

ml_res_pdb["pdbid"] = ml_res_pdb['pdbid_chain'].str.split('_').str[0]

pdbids_with_IgC1 = ml_res_pdb[ml_res_pdb['Top_Predicted_Label'] == 'IgC1']['pdbid'].unique()

# Step 3: Keep all rows from df where pdbid is in the above list
filtered_df_with_IgC1 = ml_res_pdb[ml_res_pdb['pdbid'].isin(pdbids_with_IgC1)]

#grouped_proteome = igc1_df.copy()

grouped_proteome = filtered_df_with_IgC1.copy()
grouped_proteome[['start', 'end']] = grouped_proteome['igdomain_res_range'].str.split('_', expand=True).astype(int)

# Concatenating 'start' and 'end' columns into a single 'range' column
grouped_proteome['igres_range'] = grouped_proteome['start'].astype(str) + '_' + grouped_proteome['end'].astype(str)

# Sorting the dataframe based on the starting values of the ranges
grouped_proteome = grouped_proteome.sort_values(by=['start', 'end'])
# Grouping by 'group_column' and aggregating ranges and corresponding values into a list
grouped_proteome_final = grouped_proteome.groupby('pdbid_chain').agg({
'igres_range': list,
'Top_Predicted_Label': list,
'Probability':list,
'score': list,
'refpdbname': list,
'seqid': list,
'uniprot_gene': 'first',
'cd_number': 'first',
'description':'first',
"pdbid": 'first',
'link': 'first',
'icn3dlink': 'first' 

}).reset_index()

print("__",grouped_proteome_final.shape)

#grouped_proteome_final.to_excel("IgC1_unique_genes_all.xlsx", index=False)
#grouped_proteome_final.to_excel("IgC1_containing_all_igdomains_unique_genes.xlsx", index=False)



# Convert 'pdbid' from 'pdbid_chain'
grouped_proteome_final['pdbid'] = grouped_proteome_final['pdbid_chain'].str.split('_').str[0].astype(str)

# Convert list in Top_Predicted_Label to tuple for grouping
grouped_proteome_final['Top_Predicted_Label_tuple'] = grouped_proteome_final['Top_Predicted_Label'].apply(tuple)

# Group by the tuple version
label_summary = grouped_proteome_final.groupby('Top_Predicted_Label_tuple').agg(
    count=('pdbid_chain', 'count'),
    pdbids=('pdbid', lambda x: sorted(set(x)))
).reset_index()

# Optional: convert tuple back to list for readability
label_summary['Top_Predicted_Label'] = label_summary['Top_Predicted_Label_tuple'].apply(list)
label_summary.drop(columns='Top_Predicted_Label_tuple', inplace=True)

# Save to Excel
label_summary.to_excel("chain_igC1_grouped_list.xlsx", index=False)