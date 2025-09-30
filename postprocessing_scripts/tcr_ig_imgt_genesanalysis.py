import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd

def analyze_igtype_arch_by_gene_prefix(df, prefixes, label=None):
    """
    Analyzes predicted_igtype_arch and tm_igdomain_arch for genes starting with given prefixes.

    Parameters:
    - df: pandas DataFrame with columns 'gene', 'predicted_igtype_arch', 'tm_igdomain_arch'
    - prefixes: list of gene prefixes to filter on (e.g., ['IGKV', 'IGLV'])
    - label: optional name to describe the group (for printing)

    Returns:
    - Combined distribution DataFrame
    """
    filtered = df[df['gene_name'].fillna('').str.startswith(tuple(prefixes))]


    gene_count = filtered['gene_name'].nunique()
    label_text = f"{label} " if label else ""
    print(f"Number of unique {label_text}genes starting with {prefixes}: {gene_count}")

    pred_dist = filtered['predicted_igtype_arch'].value_counts()
    tm_dist = filtered['tm_igtype_arch'].value_counts()
    refpdb_dist = filtered['refpdbname_arch'].value_counts()

    print(f"\nDistribution of predicted_igtype_arch ({label_text.strip()}):")
    print(pred_dist)

    print(f"\nDistribution of tm_igdomain_arch ({label_text.strip()}):")
    print(tm_dist)

    print(f"\nDistribution of refpdbname_arch ({label_text.strip()}):")
    print(refpdb_dist)

    combined = pd.DataFrame({
        'predicted_igtype_arch': pred_dist,
        'tm_igdomain_arch': tm_dist
    }).fillna(0).astype(int)

    print(f"\nCombined distribution ({label_text.strip()}):")
    print(combined)

    return combined



df = pd.read_excel("chain_architecture_all_unfiltered.xlsx")

# Analyze BCR genes
bcr_prefixes = ['IGKV', 'IGLV', 'IGHV']
bcr_result = analyze_igtype_arch_by_gene_prefix(df, bcr_prefixes, label='BCR')

#bcr heavy

bcr_prefixes = ['IGHV']
bcr_result = analyze_igtype_arch_by_gene_prefix(df, bcr_prefixes, label='BCR_heavy')

# bcr Lamda

bcr_prefixes = ['IGLV']
bcr_result = analyze_igtype_arch_by_gene_prefix(df, bcr_prefixes, label='BCR_lamdalight')

# Analyze TCR genes
tcr_prefixes = ['TRAV', 'TRBV', 'TRDV', 'TRGV']
tcr_result = analyze_igtype_arch_by_gene_prefix(df, tcr_prefixes, label='TCR')

#  Analyze TCR genes alpha
tcr_alpha_prefixes = ['TRAV']
tcr_result = analyze_igtype_arch_by_gene_prefix(df, tcr_alpha_prefixes, label='TCR_alpha')


# TCR constant 
tcr_constant = ['TRAC', 'TRBC', 'TRDC', 'TRGC']
tcr_constant_result = analyze_igtype_arch_by_gene_prefix(df, tcr_constant, label='tcr_constant')


# TCR constant 
tcr_constant_alpha = ['TRAC']
tcr_constant_alpha_result = analyze_igtype_arch_by_gene_prefix(df, tcr_constant_alpha , label='tcr_constant_alpha ')


# BCR constant
# NOte in unipro IGHM starting have IGHMBP2 which is not BCR so I removed discard. 

bcr_constant = ['IGHM', 'IGHD', 'IGHG', 'IGHE', 'IGHA', 'IGKC', 'IGLC']
#bcr_constant = ['IGHA', 'IGLC']

#bcr_constant = ['IGHM']

#bcr_constant = [ 'IGHG',  'IGKC']

bcr_constant_result = analyze_igtype_arch_by_gene_prefix(df, bcr_constant, label='bcr_constant')

