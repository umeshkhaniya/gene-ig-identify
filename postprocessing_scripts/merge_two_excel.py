import pandas as pd
import uniprot_data_parse 
import cdd_igdomain_igtype

# Here I combine ML prediction, uniprot annotation data.

# Step 1: Load the Excel sheets into DataFrames
df1 = pd.read_excel('human_unique_proteome_all_0.4.xlsx')
print(list(df1.columns))

# let add the uniprot annotation

# Extract UniProt ID
df1['uniprotid'] = df1['id_chain'].str.split('_').str[0]

df1['icn3d_link'] = df1['uniprotid'].apply(
    lambda uniprotid: f'=HYPERLINK("https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid={uniprotid}&command=ig+refnum+on", "view_structure")'
)

df1['uniprot_link'] = df1['uniprotid'].apply(
    lambda uniprotid: f'=HYPERLINK("https://www.uniprot.org/uniprotkb/{uniprotid}/entry", "view_uniprot")'
)


uniprot_annotation  = uniprot_data_parse.parse_uniprot("../../../input/data_from_uniprot_browser/all_human_proteomes_83587_Jun18_2025/uniprotkb_proteome_UP000005640_2025_06_18.json")
#uniprot_parse_data = parse_uniprot("../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/sample1.json")


#uniprot_annotation = uniprot_data_parse.parse_uniprot("../../input/data_from_uniprot_browser/all_human_proteome_82493_april12_2024/sample1.json")
# Expand the dictionary into a DataFrame
dict_uniprot_annotation = pd.DataFrame.from_dict(uniprot_annotation, orient='index').reset_index().rename(columns={'index': 'uniprotid'})

# Merge DataFrame with dictionary values
df1 = df1.merge(dict_uniprot_annotation, on='uniprotid', how='left')
# Drop uniprot_id if not needed
df1 = df1.drop(columns='uniprotid')

df2 = pd.read_excel('top_predictions_uniquegenes.xlsx')

# Step 2: Create a new column in df1 by combining the relevant columns
df1['combined'] = df1['id_chain'] + '_' + df1['igdomain_res_range'] +  '_' + df1['refpdbname'] + '_' + df1['ig_type']

# Step 3: Merge df1 and df2 based on 'combined' in df1 and 'Graph' in df2
merged_df1 = pd.merge(df1, df2, left_on='combined', right_on='Graph', how='inner')

print(list(merged_df1.columns))




# add CDD annotation
merged_df = cdd_igdomain_igtype.add_cdd_annotation_column(merged_df1)

# List of columns you want first
first_cols = [
    'id_chain', 'igdomain_res_range', 'score', 'ig_type', 
    'predicted_igtype', 'probability', 'seqid',
    'refpdbname', 'uniprot_domain_resrange', 'icn3d_link', 'uniprot_link', 'description']

remove_column = ["Graph", "combined", "link"]

# Get the rest of the columns that are not in first_cols
other_cols = [col for col in merged_df.columns if col not in first_cols + remove_column]

merged_df  = merged_df[first_cols + other_cols]

# rename ig_type to tm_ig_type
# score to tm_score

# Rename only the specific columns
merged_df = merged_df.rename(columns={
    "score": "tm_score",
    "ig_type": "tm_igtype",
    "uniprot_gene": "gene_name"
})


# Step 6: Save the final DataFrame to an Excel file
#merged_df.to_excel('merged_result_uniquegenes.xlsx', index=False)

merged_df.to_excel('merged1_result_uniquegenes_human.xlsx', index=False)
