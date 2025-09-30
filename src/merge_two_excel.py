import pandas as pd
import uniprot_data_parse 

# Here I combine ML prediction, uniprot annotation data.

# Step 1: Load the Excel sheets into DataFrames
df1 = pd.read_excel('human_proteome_all_igstrand.xlsx')

# let add the uniprot annotation

# Extract UniProt ID
df1['uniprotid'] = df1['pdbid_chain'].str.split('_').str[0]
df1['icn3dlink'] = df1['uniprotid'].apply(lambda uniprotid: f"https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={uniprotid}")



uniprot_annotation  = uniprot_data_parse.parse_uniprot("../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/uniprotkb_proteome_UP000005640_2024_04_12.json")
#uniprot_parse_data = parse_uniprot("../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/sample1.json")


#uniprot_annotation = uniprot_data_parse.parse_uniprot("../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/sample1.json")
# Expand the dictionary into a DataFrame
dict_uniprot_annotation = pd.DataFrame.from_dict(uniprot_annotation, orient='index').reset_index().rename(columns={'index': 'uniprotid'})

# Merge DataFrame with dictionary values
df1 = df1.merge(dict_uniprot_annotation, on='uniprotid', how='left')
# Drop uniprot_id if not needed
df1 = df1.drop(columns='uniprotid')

df2 = pd.read_excel('top_predictions.xlsx')

# Step 2: Create a new column in df1 by combining the relevant columns
df1['combined'] = df1['pdbid_chain'] + '_' + df1['igdomain_res_range'] +  '_' + df1['refpdbname'] + '_' + df1['ig_type']

# Step 3: Merge df1 and df2 based on 'combined' in df1 and 'Graph' in df2
merged_df = pd.merge(df1, df2, left_on='combined', right_on='Graph', how='inner')


# List of columns you want first
first_cols = [
    'pdbid_chain', 'igdomain_res_range', 'score', 'ig_type',
    'Top_Predicted_Label', 'Probability', 'seqid',
    'refpdbname', 'uniprot_domain_resrange', 'icn3dlink', 'description']

remove_column = ["Graph", "combined"]

# Get the rest of the columns that are not in first_cols
other_cols = [col for col in merged_df.columns if col not in first_cols + remove_column]

merged_df  = merged_df[first_cols + other_cols]


# Step 6: Save the final DataFrame to an Excel file
merged_df.to_excel('merged_result1.xlsx', index=False)
